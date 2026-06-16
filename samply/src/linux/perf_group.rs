use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::os::unix::io::RawFd;
use std::time::Duration;
use std::{fs, io};

use byteorder::LittleEndian;
use linux_perf_data::linux_perf_event_reader::get_record_timestamp;
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};

use super::perf_event::{EventRef, EventSource, ExtraEvent, Perf};
use super::sorter::EventSorter;

struct StoppedProcess(u32);

impl StoppedProcess {
    fn new(pid: u32) -> Result<Self, io::Error> {
        // debug!("Stopping process with PID {}...", pid);
        let ok = unsafe { libc::kill(pid as _, libc::SIGSTOP) };
        if ok < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(StoppedProcess(pid))
    }
}

impl Drop for StoppedProcess {
    fn drop(&mut self) {
        // debug!("Resuming process with PID {}...", self.0);
        unsafe {
            libc::kill(self.0 as _, libc::SIGCONT);
        }
    }
}

/// One sampling perf event with its ring buffer.
struct Member {
    perf: Perf,
    is_closed: bool,
}

impl Member {
    fn new(perf: Perf) -> Self {
        Member {
            perf,
            is_closed: false,
        }
    }
}

impl Deref for Member {
    type Target = Perf;
    fn deref(&self) -> &Self::Target {
        &self.perf
    }
}

impl DerefMut for Member {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.perf
    }
}

/// The perf events opened for one profiling session, together with the
/// machinery to poll their ring buffers and merge the records into one
/// time-ordered stream.
///
/// Despite the name, this is not a kernel event group: its members are
/// independent sampling events.
pub struct PerfGroup {
    event_sorter: EventSorter<RawFd, u64, EventRef>,
    /// One sampling event per ring buffer, keyed by fd.
    members: BTreeMap<RawFd, Member>,
    poll: Poll,
    poll_events: Events,
    frequency: u32,
    stack_size: u32,
    regs_mask: u64,
    event_source: EventSource,
    extra_events: Vec<ExtraEvent>,
    stopped_processes: Vec<StoppedProcess>,
    /// When true, the extra-event group can't be opened with `inherit` on this
    /// kernel (pre-6.12), so events are opened per-CPU system-wide (no inherit)
    /// and samples must be filtered to [`Self::root_pids`] downstream.
    system_wide: bool,
    /// The pids `open_process` was asked to profile; the roots of the process
    /// trees whose samples we keep when running system-wide.
    root_pids: Vec<u32>,
}

/// Every online CPU on the machine, parsed from `/sys/devices/system/cpu/online`
/// (a comma-separated list of ids and ranges, e.g. `"0-3,5,8-11"`).
///
/// `num_cpus::get()` would be wrong here: it counts only the CPUs in the
/// caller's cpuset, but we need the CPUs the *profiled* process can run on, and
/// that process may be confined to a disjoint cpuset we can't observe from here.
/// Falls back to `0..num_cpus::get()` if sysfs is unreadable.
fn online_cpus() -> Vec<u32> {
    fn parse(list: &str) -> Option<Vec<u32>> {
        let mut cpus = Vec::new();
        for part in list.trim().split(',') {
            if part.is_empty() {
                continue;
            }
            match part.split_once('-') {
                Some((start, end)) => {
                    let start: u32 = start.trim().parse().ok()?;
                    let end: u32 = end.trim().parse().ok()?;
                    cpus.extend(start..=end);
                }
                None => cpus.push(part.trim().parse().ok()?),
            }
        }
        (!cpus.is_empty()).then_some(cpus)
    }

    fs::read_to_string("/sys/devices/system/cpu/online")
        .ok()
        .and_then(|s| parse(&s))
        .unwrap_or_else(|| (0..num_cpus::get() as u32).collect())
}

fn get_threads(pid: u32) -> Result<Vec<u32>, io::Error> {
    let entries = fs::read_dir(format!("/proc/{pid}/task"))?;
    let tids = entries
        .flatten()
        .filter_map(|entry| {
            let tid: u32 = entry.file_name().to_string_lossy().parse().unwrap();
            if tid != pid {
                Some(tid)
            } else {
                None
            }
        })
        .collect();
    Ok(tids)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachMode {
    AttachWithEnableOnExec,
    StopAttachEnableResume,
}

impl PerfGroup {
    pub fn new(
        frequency: u32,
        stack_size: u32,
        regs_mask: u64,
        event_source: EventSource,
        extra_events: Vec<ExtraEvent>,
    ) -> Self {
        // Extra events ride along on each sample via PERF_SAMPLE_READ. On
        // kernels that reject `inherit` + PERF_SAMPLE_READ (pre-6.12), fall
        // back to per-CPU system-wide capture, which follows every thread on
        // the CPU without `inherit`.
        let system_wide = !extra_events.is_empty() && !Perf::supports_inherited_sample_read();
        if system_wide {
            eprintln!(
                "Note: this kernel can't attach the extra perf events to inherited samples; \
                 capturing them per-CPU system-wide and filtering to the launched process tree."
            );
        }
        PerfGroup {
            event_sorter: EventSorter::new(),
            members: Default::default(),
            poll: Poll::new().unwrap(),
            poll_events: Events::with_capacity(16),
            frequency,
            stack_size,
            event_source,
            extra_events,
            regs_mask,
            stopped_processes: Vec::new(),
            system_wide,
            root_pids: Vec::new(),
        }
    }

    /// Whether events are opened per-CPU system-wide (the pre-6.12 fallback);
    /// callers must then filter samples to [`Self::root_pids`].
    pub fn is_system_wide(&self) -> bool {
        self.system_wide
    }

    /// The pids whose process trees this group is profiling.
    pub fn root_pids(&self) -> &[u32] {
        &self.root_pids
    }

    pub fn open(
        pid: u32,
        frequency: u32,
        stack_size: u32,
        event_source: EventSource,
        regs_mask: u64,
        attach_mode: AttachMode,
        extra_events: Vec<ExtraEvent>,
    ) -> Result<Self, io::Error> {
        let mut group =
            PerfGroup::new(frequency, stack_size, regs_mask, event_source, extra_events);
        group.open_process(pid, attach_mode)?;
        Ok(group)
    }

    pub fn open_process(&mut self, pid: u32, attach_mode: AttachMode) -> Result<(), io::Error> {
        if attach_mode == AttachMode::StopAttachEnableResume {
            self.stopped_processes.push(StoppedProcess::new(pid)?);
        }
        self.root_pids.push(pid);
        let system_wide = self.system_wide;
        let mut perf_events = Vec::new();
        let threads = get_threads(pid)?;

        let open_perf = |pid: u32, cpu: Option<u32>| -> io::Result<Perf> {
            let mut builder = Perf::build()
                .pid(pid)
                .frequency(self.frequency as u64)
                .sample_user_stack(self.stack_size)
                .sample_user_regs(self.regs_mask)
                .gather_context_switches()
                .event_source(self.event_source)
                .extra_events(self.extra_events.clone())
                .start_disabled();
            if let Some(cpu) = cpu {
                builder = builder.only_cpu(cpu);
                if system_wide {
                    // System-wide on this CPU: captures every thread scheduled
                    // here (including ones spawned later) without `inherit`,
                    // which is what makes PERF_SAMPLE_READ legal pre-6.12.
                    builder = builder.all_pids();
                } else {
                    builder = builder.inherit_to_children();
                }
            } else {
                builder = builder.any_cpu();
            }
            // ENABLE_ON_EXEC only fires for events attached to the task that
            // execs. System-wide events aren't, so they're enabled explicitly
            // by the caller instead (see `init_profiler`).
            if attach_mode == AttachMode::AttachWithEnableOnExec && !system_wide {
                builder = builder.enable_on_exec();
            }
            builder.open()
        };

        // A per-CPU event only ever sees the CPU it was opened on, so we must
        // open one for every CPU the target might be scheduled on. The target's
        // cpuset can be disjoint from ours and unknowable at this point: e.g.
        // under CodSpeed the target is launched into a separate cgroup slice via
        // `systemd-run`, so by the time it exists and is pinned we've long since
        // opened these events. Covering all online CPUs sidesteps that entirely.
        let cpu_ids = online_cpus();
        let cpu_count = cpu_ids.len();
        for &cpu in &cpu_ids {
            let perf = open_perf(pid, Some(cpu))?;
            perf_events.push((Some(cpu), perf));
        }

        // System-wide per-CPU events already cover every thread on every CPU,
        // so the per-thread enrollment below (which exists to attach to the
        // already-running threads of the target pid) is only needed in the
        // per-task + inherit mode.
        if !system_wide {
            if cpu_count * (threads.len() + 1) >= 1000 {
                for &tid in &threads {
                    let perf = open_perf(tid, None)?;
                    perf_events.push((None, perf));
                }
            } else {
                for &cpu in &cpu_ids {
                    for &tid in &threads {
                        let perf = open_perf(tid, Some(cpu))?;
                        perf_events.push((Some(cpu), perf));
                    }
                }
            }
        }

        for (_cpu, perf) in perf_events {
            let fd = perf.fd();
            self.members.insert(fd, Member::new(perf));
            self.poll.registry().register(
                &mut SourceFd(&fd),
                Token(fd as usize),
                Interest::READABLE,
            )?;
        }

        Ok(())
    }

    /// The (kernel-assigned id, name) of every extra event, across all
    /// members. Each member's kernel event group has its own instances, so
    /// several ids share a name.
    pub fn extra_event_ids_and_names(&self) -> Vec<(u64, String)> {
        self.members
            .values()
            .flat_map(|member| {
                member
                    .extra_event_ids()
                    .map(|(id, name)| (id, name.to_owned()))
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn enable(&mut self) {
        for perf in self.members.values_mut() {
            perf.enable();
        }

        self.stopped_processes.clear();
    }

    pub fn wait(&mut self) {
        for member in self.members.values() {
            if member.are_events_pending() {
                return;
            }
        }

        let result = self
            .poll
            .poll(&mut self.poll_events, Some(Duration::from_millis(100)));
        if let Err(err) = result {
            eprintln!("poll failed: {err}");
            return;
        }

        for ev in self.poll_events.iter() {
            if ev.is_read_closed() {
                let fd = ev.token().0 as RawFd;
                self.members.get_mut(&fd).unwrap().is_closed = true;
            }
        }
    }

    pub fn consume_events(&mut self, mut cb: impl FnMut(EventRef)) {
        let mut fds_to_remove = Vec::new();
        loop {
            for (&fd, member) in &mut self.members {
                self.event_sorter.begin_group(fd);
                while let Some(ev) = self.event_sorter.pop() {
                    cb(ev);
                }

                let perf = &mut member.perf;
                if !perf.are_events_pending() {
                    if member.is_closed {
                        fds_to_remove.push(perf.fd());
                        continue;
                    }
                    continue;
                }

                self.event_sorter.extend(perf.iter().map(|event| {
                    let rec = event.get();
                    let timestamp = get_record_timestamp::<LittleEndian>(
                        rec.record_type,
                        rec.data,
                        &rec.parse_info,
                    )
                    .expect("All events should have a record identifier");
                    (timestamp, event)
                }));
            }

            self.event_sorter.advance_round();
            while let Some(ev) = self.event_sorter.pop() {
                cb(ev);
            }

            for fd in fds_to_remove.drain(..) {
                let result = self.poll.registry().deregister(&mut SourceFd(&fd));
                if let Err(err) = result {
                    eprintln!("deregister failed: {err}");
                    continue;
                }
                self.members.remove(&fd);
            }

            if !self.event_sorter.has_more() {
                break;
            }
        }
    }

    /// Consume any remaining events from ring buffers, then flush all events
    /// still held back in the sorter. Call this before finishing the profile.
    pub fn flush_events(&mut self, mut cb: impl FnMut(EventRef)) {
        self.consume_events(&mut cb);
        while let Some(ev) = self.event_sorter.force_pop() {
            cb(ev);
        }
    }
}
