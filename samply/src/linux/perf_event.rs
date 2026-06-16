use std::cell::RefCell;
use std::cmp::max;
use std::collections::BinaryHeap;
use std::ops::Range;
use std::os::unix::io::RawFd;
use std::rc::Rc;
use std::sync::atomic::{fence, Ordering};
use std::{cmp, fmt, io, mem, ptr, slice};

use libc::{self, c_int, c_void, pid_t};
use linux_perf_data::linux_perf_event_reader;
use linux_perf_event_reader::{Endianness, RawData, RawEventRecord, RecordParseInfo, RecordType};

use super::sys::*;

#[derive(Debug)]
#[repr(C)]
struct PerfEventHeader {
    kind: u32,
    misc: u16,
    size: u16,
}

#[derive(Clone, Debug)]
enum SliceLocation {
    Single(Range<usize>),
    Split(Range<usize>, Range<usize>),
}

impl SliceLocation {
    #[inline]
    fn get<'a>(&self, buffer: &'a [u8]) -> RawData<'a> {
        match *self {
            SliceLocation::Single(ref range) => RawData::Single(&buffer[range.clone()]),
            SliceLocation::Split(ref left, ref right) => {
                RawData::Split(&buffer[left.clone()], &buffer[right.clone()])
            }
        }
    }
}

#[derive(Clone, Debug)]
struct RawRecordLocation {
    kind: u32,
    misc: u16,
    data_location: SliceLocation,
}

impl RawRecordLocation {
    #[inline]
    fn get<'a>(&self, buffer: &'a [u8], parse_info: RecordParseInfo) -> RawEventRecord<'a> {
        RawEventRecord {
            record_type: RecordType(self.kind),
            misc: self.misc,
            data: self.data_location.get(buffer),
            parse_info,
        }
    }
}

unsafe fn read_head(pointer: *const u8) -> u64 {
    let page = &*(pointer as *const PerfEventMmapPage);
    let head = ptr::read_volatile(&page.data_head);
    fence(Ordering::Acquire);
    head
}

unsafe fn read_tail(pointer: *const u8) -> u64 {
    let page = &*(pointer as *const PerfEventMmapPage);
    // No memory fence required because we're just reading a value previously
    // written by us.
    ptr::read_volatile(&page.data_tail)
}

unsafe fn write_tail(pointer: *mut u8, value: u64) {
    let page = &mut *(pointer as *mut PerfEventMmapPage);
    fence(Ordering::AcqRel);
    ptr::write_volatile(&mut page.data_tail, value);
}

/// An extra event opened as a sibling of the sampling event in its kernel
/// event group. Siblings don't sample and have no ring buffer of their own;
/// they are scheduled onto the PMU together with the leader, and their values
/// are delivered inline in the leader's samples via `PERF_SAMPLE_READ`.
///
/// The id/name pairing is established here, at open time, because the fd is
/// the only link between the kernel-assigned id and the event we asked for.
#[derive(Debug)]
struct SiblingEvent {
    fd: RawFd,
    /// The kernel-assigned id of this event, matching `ReadValue::id` in the
    /// leader's samples.
    id: u64,
    /// Label for the counter track this event's values feed.
    name: String,
}

/// One sampling perf event with its ring buffer.
#[derive(Debug)]
pub struct Perf {
    event_ref_state: Rc<RefCell<EventRefState>>,
    buffer: *mut u8,
    size: u64,
    fd: RawFd,
    /// Sibling events of this event's kernel event group. Empty unless extra
    /// events were requested.
    siblings: Vec<SiblingEvent>,
    /// The leader's own kernel-assigned id and the name it was requested
    /// under, when one of the extra events has the same encoding as the
    /// sampling event itself (e.g. `cpu-cycles` on the cycles leader). The
    /// leader's value is part of every group read anyway, so such an event
    /// needs no sibling — and no PMU counter — of its own.
    leader_alias: Option<(u64, String)>,
    position: u64,
    parse_info: RecordParseInfo,
}

impl Drop for Perf {
    fn drop(&mut self) {
        unsafe {
            for sibling in &self.siblings {
                libc::close(sibling.fd);
            }
            libc::close(self.fd);
        }
    }
}

#[inline]
unsafe fn get_buffer<'a>(buffer: *const u8, size: u64) -> &'a [u8] {
    slice::from_raw_parts(buffer.offset(4096), size as usize)
}

fn next_raw_event(
    buffer: *const u8,
    size: u64,
    position_cell: &mut u64,
) -> Option<RawRecordLocation> {
    let head = unsafe { read_head(buffer) };
    if head == *position_cell {
        return None;
    }

    let buffer = unsafe { get_buffer(buffer, size) };
    let position = *position_cell;
    let relative_position = position % size;
    let event_position = relative_position as usize;
    let event_data_position =
        (relative_position + mem::size_of::<PerfEventHeader>() as u64) as usize;
    let event_header = unsafe {
        &*(&buffer[event_position..event_data_position] as *const _ as *const PerfEventHeader)
    };
    let next_event_position = event_position + event_header.size as usize;

    let data_location = if next_event_position > size as usize {
        let first = event_data_position..buffer.len();
        let second = 0..next_event_position % size as usize;
        SliceLocation::Split(first, second)
    } else {
        SliceLocation::Single(event_data_position..next_event_position)
    };

    let raw_event_location = RawRecordLocation {
        kind: event_header.kind,
        misc: event_header.misc,
        data_location,
    };

    // trace!("Parsed raw event: {:?}", raw_event_location);

    let next_position = position + event_header.size as u64;
    *position_cell = next_position;

    Some(raw_event_location)
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum EventSource {
    HwCpuCycles,
    SwCpuClock,
}

impl EventSource {
    /// The `(perf_event_attr.type, perf_event_attr.config)` pair for this event.
    fn type_and_config(self) -> (u32, u64) {
        match self {
            EventSource::HwCpuCycles => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
            EventSource::SwCpuClock => (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK),
        }
    }
}

/// An extra counter to read alongside the main sampling event.
///
/// Extra events are opened as counting-only siblings of the sampling event in
/// its kernel event group. With `PERF_SAMPLE_READ | PERF_FORMAT_GROUP`, every
/// sample carries the current value of each sibling, so the per-sample delta
/// gives the number of events that occurred since the previous sample.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExtraEvent {
    /// `perf_event_attr.type` (e.g. `PERF_TYPE_HARDWARE`, `PERF_TYPE_RAW`).
    pub event_type: u32,
    /// `perf_event_attr.config`.
    pub config: u64,
    /// A human-readable name, used to label the resulting counter track.
    pub name: String,
}

impl std::str::FromStr for ExtraEvent {
    type Err = String;

    /// Parse a `<name>:<type>:<config>` event spec, e.g. `instructions:0:0x1`
    /// or `l1d_access:4:0x0729`. `<type>` and `<config>` go verbatim into
    /// `perf_event_attr.type` / `.config` (decimal or `0x`-prefixed hex);
    /// resolving an event name to its encoding is the caller's job. `<name>`
    /// labels the resulting counter column.
    fn from_str(spec: &str) -> Result<Self, Self::Err> {
        let err = || format!("expected '<name>:<type>:<config>', got '{spec}'");
        let (rest, config) = spec.rsplit_once(':').ok_or_else(err)?;
        let (name, event_type) = rest.rsplit_once(':').ok_or_else(err)?;
        if name.is_empty() {
            return Err(err());
        }
        Ok(ExtraEvent {
            event_type: parse_u32(event_type).ok_or_else(err)?,
            config: parse_u64(config).ok_or_else(err)?,
            name: name.to_owned(),
        })
    }
}

fn parse_u64(value: &str) -> Option<u64> {
    match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => value.parse().ok(),
    }
}

fn parse_u32(value: &str) -> Option<u32> {
    parse_u64(value)?.try_into().ok()
}

impl ExtraEvent {
    /// The `perf_event_attr` for opening this event as a counting-only sibling
    /// in the group of a leader opened with `leader_attr`.
    fn sibling_attr(&self, leader_attr: &PerfEventAttr) -> PerfEventAttr {
        let mut attr: PerfEventAttr = unsafe { mem::zeroed() };
        attr.size = mem::size_of::<PerfEventAttr>() as u32;
        attr.kind = self.event_type;
        attr.config = self.config;
        // The sibling must use the same read_format as the leader so the
        // grouped read layout is consistent across the group, and must match
        // the leader's clock and inherit setting, otherwise perf_event_open
        // rejects it with EINVAL. Counting only: sample_period_or_freq stays
        // 0, so the sibling produces no records of its own and needs no ring
        // buffer.
        attr.read_format = leader_attr.read_format;
        attr.clock_id = leader_attr.clock_id;
        // Siblings are opened enabled, but a group is only scheduled onto the
        // PMU while its leader is enabled, so the leader's disabled /
        // enable_on_exec state gates the whole group. (Opening siblings
        // disabled would be wrong: PERF_EVENT_IOC_ENABLE on the leader only
        // enables the leader, and would leave the siblings off forever.)
        //
        // USE_CLOCKID is copied alongside clock_id above: the kernel requires
        // every event in a group to use the same clock, and rejects a sibling
        // that carries the leader's clock_id without also setting the flag.
        attr.flags = leader_attr.flags
            & (PERF_ATTR_FLAG_EXCLUDE_KERNEL
                | PERF_ATTR_FLAG_INHERIT
                | PERF_ATTR_FLAG_USE_CLOCKID);
        attr
    }
}

/// Open `extra` as a counting-only sibling in `leader_fd`'s event group and
/// read back its kernel-assigned id.
fn open_sibling_event(
    extra: &ExtraEvent,
    leader_attr: &PerfEventAttr,
    pid: pid_t,
    cpu: c_int,
    leader_fd: RawFd,
) -> io::Result<SiblingEvent> {
    let attr = extra.sibling_attr(leader_attr);
    let fd = sys_perf_event_open(&attr, pid, cpu, leader_fd, PERF_FLAG_FD_CLOEXEC);
    if fd == -1 {
        return Err(io::Error::last_os_error());
    }

    match read_event_id(fd) {
        Ok(id) => Ok(SiblingEvent {
            fd,
            id,
            name: extra.name.clone(),
        }),
        Err(err) => {
            unsafe {
                libc::close(fd);
            }
            Err(err)
        }
    }
}

/// Read the kernel-assigned id of the event behind `fd`.
fn read_event_id(fd: RawFd) -> io::Result<u64> {
    let mut id: u64 = 0;
    let ok = unsafe { libc::ioctl(fd, PERF_EVENT_IOC_ID as _, &mut id) };
    if ok == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(id)
}

#[derive(Clone, Debug)]
pub struct PerfBuilder {
    pid: u32,
    /// Open the event system-wide (`pid = -1`), capturing every thread
    /// scheduled on the selected CPU instead of a single task tree. Used as a
    /// fallback when the kernel can't combine `inherit` with
    /// `PERF_SAMPLE_READ` (see [`Perf::supports_inherited_sample_read`]).
    all_processes: bool,
    cpu: Option<u32>,
    frequency: u64,
    stack_size: u32,
    reg_mask: u64,
    event_source: EventSource,
    extra_events: Vec<ExtraEvent>,
    inherit: bool,
    start_disabled: bool,
    enable_on_exec: bool,
    exclude_kernel: bool,
    gather_context_switches: bool,
}

impl PerfBuilder {
    pub fn pid(mut self, pid: u32) -> Self {
        self.pid = pid;
        self
    }

    /// Open the event system-wide (`pid = -1`) rather than scoped to a single
    /// pid. Must be paired with [`Self::only_cpu`] (a system-wide event needs a
    /// CPU) and is mutually exclusive with [`Self::inherit_to_children`].
    pub fn all_pids(mut self) -> Self {
        self.all_processes = true;
        self
    }

    pub fn only_cpu(mut self, cpu: u32) -> Self {
        self.cpu = Some(cpu);
        self
    }

    pub fn any_cpu(mut self) -> Self {
        self.cpu = None;
        self
    }

    pub fn frequency(mut self, frequency: u64) -> Self {
        self.frequency = frequency;
        self
    }

    pub fn sample_user_stack(mut self, stack_size: u32) -> Self {
        self.stack_size = stack_size;
        self
    }

    pub fn sample_user_regs(mut self, reg_mask: u64) -> Self {
        self.reg_mask = reg_mask;
        self
    }

    /// Turns on the kernel measurements. This requires the `/proc/sys/kernel/perf_event_paranoid` to be less than `2`.
    pub fn sample_kernel(mut self) -> Self {
        self.exclude_kernel = false;
        self
    }

    pub fn event_source(mut self, event_source: EventSource) -> Self {
        self.event_source = event_source;
        self
    }

    /// Attach extra counters to be read on every sample of the main event.
    pub fn extra_events(mut self, extra_events: Vec<ExtraEvent>) -> Self {
        self.extra_events = extra_events;
        self
    }

    pub fn inherit_to_children(mut self) -> Self {
        self.inherit = true;
        self
    }

    pub fn start_disabled(mut self) -> Self {
        self.start_disabled = true;
        self
    }

    pub fn enable_on_exec(mut self) -> Self {
        self.enable_on_exec = true;
        self
    }

    pub fn gather_context_switches(mut self) -> Self {
        self.gather_context_switches = true;
        self
    }

    pub fn open(self) -> io::Result<Perf> {
        // `-1` means "all processes" (system-wide), which the kernel only
        // accepts together with a specific CPU.
        let pid: i32 = if self.all_processes { -1 } else { self.pid as i32 };
        let cpu = self.cpu.map(|cpu| cpu as i32).unwrap_or(-1);
        let frequency = self.frequency;
        let stack_size = self.stack_size;
        let reg_mask = self.reg_mask;
        let event_source = self.event_source;
        let inherit = self.inherit;
        let start_disabled = self.start_disabled;
        let exclude_kernel = self.exclude_kernel;
        let gather_context_switches = self.gather_context_switches;

        // debug!(
        //     "Opening perf events; pid={}, cpu={}, frequency={}, stack_size={}, reg_mask=0x{:016X}, event_source={:?}, inherit={}, start_disabled={}...",
        //     pid,
        //     cpu,
        //     frequency,
        //     stack_size,
        //     reg_mask,
        //     event_source,
        //     inherit,
        //     start_disabled
        // );

        let max_sample_rate = Perf::max_sample_rate();
        if let Some(max_sample_rate) = max_sample_rate {
            // debug!("Maximum sample rate: {}", max_sample_rate);
            if frequency > max_sample_rate {
                let message = format!( "frequency can be at most {max_sample_rate} as configured in /proc/sys/kernel/perf_event_max_sample_rate" );
                return Err(io::Error::new(io::ErrorKind::InvalidInput, message));
            }
        }

        if stack_size > 63 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sample_user_stack can be at most 63kb",
            ));
        }

        // See `perf_mmap` in the Linux kernel.
        if cpu == -1 && inherit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "you can't inherit to children and run on all cpus at the same time",
            ));
        }

        assert_eq!(mem::size_of::<PerfEventMmapPage>(), 1088);

        if cfg!(target_arch = "x86_64") {
            assert_eq!(PERF_EVENT_IOC_ENABLE, 9216);
        } else if cfg!(target_arch = "mips64") {
            assert_eq!(PERF_EVENT_IOC_ENABLE, 536880128);
        }

        let mut attr: PerfEventAttr = unsafe { mem::zeroed() };
        attr.size = mem::size_of::<PerfEventAttr>() as u32;

        let (event_type, event_config) = event_source.type_and_config();
        attr.kind = event_type;
        attr.config = event_config;

        attr.sample_type = PERF_SAMPLE_IP
            | PERF_SAMPLE_TID
            | PERF_SAMPLE_TIME
            | PERF_SAMPLE_CPU
            | PERF_SAMPLE_PERIOD;

        if reg_mask != 0 {
            attr.sample_type |= PERF_SAMPLE_REGS_USER;
        }

        if stack_size != 0 {
            attr.sample_type |= PERF_SAMPLE_STACK_USER;
        }

        // When extra events are requested, make this the leader of a kernel
        // event group and have every sample carry the group's counter values,
        // each tagged with its event's kernel-assigned id.
        if !self.extra_events.is_empty() {
            attr.sample_type |= PERF_SAMPLE_READ;
            attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
        }

        // Always request callchain so the kernel walks frame pointers at sample
        // time. The user portion serves as a fallback when DWARF unwinding
        // truncates (e.g. due to the 32 KB captured stack window).
        attr.sample_type |= PERF_SAMPLE_CALLCHAIN;
        if exclude_kernel {
            attr.flags |= PERF_ATTR_FLAG_EXCLUDE_CALLCHAIN_KERNEL;
        }

        attr.sample_regs_user = reg_mask;
        attr.sample_stack_user = stack_size;
        attr.sample_period_or_freq = frequency;
        attr.clock_id = libc::CLOCK_MONOTONIC;

        attr.flags = PERF_ATTR_FLAG_DISABLED
            | PERF_ATTR_FLAG_MMAP
            | PERF_ATTR_FLAG_MMAP2
            | PERF_ATTR_FLAG_MMAP_DATA
            | PERF_ATTR_FLAG_COMM
            | PERF_ATTR_FLAG_FREQ
            | PERF_ATTR_FLAG_TASK
            | PERF_ATTR_FLAG_SAMPLE_ID_ALL
            | PERF_ATTR_FLAG_USE_CLOCKID;

        if self.enable_on_exec {
            attr.flags |= PERF_ATTR_FLAG_ENABLE_ON_EXEC;
        }

        if exclude_kernel {
            attr.flags |= PERF_ATTR_FLAG_EXCLUDE_KERNEL;
        }

        if inherit {
            attr.flags |= PERF_ATTR_FLAG_INHERIT;
        }

        if gather_context_switches {
            attr.flags |= PERF_ATTR_FLAG_CONTEX_SWITCH;
        }

        let fd = sys_perf_event_open(&attr, pid as pid_t, cpu as _, -1, PERF_FLAG_FD_CLOEXEC);
        if fd == -1 {
            let err = io::Error::last_os_error();

            // eprintln!(
            //     "The perf_event_open syscall failed for PID {}: {}",
            //     pid, err
            // );
            // if let Some(errcode) = err.raw_os_error() {
            //     if errcode == libc::EINVAL {
            //         info!("Your profiling frequency might be too high; try lowering it");
            //     }
            // }

            return Err(err);
        }

        const STACK_COUNT_PER_BUFFER: u32 = 32;
        let required_space = max(stack_size, 4096) * STACK_COUNT_PER_BUFFER;
        let page_size = 4096;
        let n = (1..26)
            .find(|n| (1_u32 << n) * 4096_u32 >= required_space)
            .expect("cannot find appropriate page count for given stack size");
        let page_count: u32 = max(1 << n, 16);
        // debug!(
        //     "Allocating {} + 1 pages for the ring buffer for PID {} on CPU {}",
        //     page_count, pid, cpu
        // );

        let full_size = (page_size * (page_count + 1)) as usize;

        let buffer;
        unsafe {
            buffer = libc::mmap(
                ptr::null_mut(),
                full_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if buffer == libc::MAP_FAILED {
                let err = io::Error::last_os_error();
                libc::close(fd);
                return Err(io::Error::new(err.kind(), format!("mmap failed: {err}")));
            }
        }

        let buffer = buffer as *mut u8;
        let size = (page_size * page_count) as u64;

        let attr_bytes_ptr = &attr as *const PerfEventAttr as *const u8;
        let attr_bytes_len = mem::size_of::<PerfEventAttr>();
        let attr_bytes = unsafe { slice::from_raw_parts(attr_bytes_ptr, attr_bytes_len) };
        let (attr2, _size) =
            linux_perf_event_reader::PerfEventAttr::parse::<_, byteorder::NativeEndian>(attr_bytes)
                .unwrap();
        let parse_info = RecordParseInfo::new(&attr2, Endianness::NATIVE);

        // Open each extra event as a counting-only sibling in the leader's
        // event group. An extra event with the same encoding as the leader is
        // aliased to the leader instead of opening a redundant sibling. If
        // anything fails, roll back the whole group so the caller can fall
        // back to a simpler configuration.
        let mut siblings: Vec<SiblingEvent> = Vec::with_capacity(self.extra_events.len());
        let mut leader_alias: Option<(u64, String)> = None;
        for extra in &self.extra_events {
            let aliases_leader = leader_alias.is_none()
                && (extra.event_type, extra.config) == (event_type, event_config);
            let result = if aliases_leader {
                read_event_id(fd).map(|id| leader_alias = Some((id, extra.name.clone())))
            } else {
                open_sibling_event(extra, &attr, pid as pid_t, cpu as _, fd)
                    .map(|sibling| siblings.push(sibling))
            };
            if let Err(err) = result {
                unsafe {
                    for sibling in &siblings {
                        libc::close(sibling.fd);
                    }
                    libc::munmap(buffer as *mut c_void, full_size);
                    libc::close(fd);
                }
                return Err(io::Error::new(
                    err.kind(),
                    format!("failed to open extra event '{}': {err}", extra.name),
                ));
            }
        }

        // debug!("Perf events open with fd={}", fd);
        let mut perf = Perf {
            event_ref_state: Rc::new(RefCell::new(EventRefState::new(buffer, size))),
            buffer,
            size,
            fd,
            siblings,
            leader_alias,
            position: 0,
            parse_info,
        };

        if !start_disabled {
            perf.enable();
        }

        Ok(perf)
    }
}

impl Perf {
    /// Whether this kernel allows opening a sampling event with both `inherit`
    /// and `PERF_SAMPLE_READ` set. Linux rejected that combination with
    /// `EINVAL` until 6.12; on such kernels the extra-event group must instead
    /// be opened system-wide and without `inherit`.
    ///
    /// Probes by opening a throwaway event that mirrors the real leader's
    /// attributes (sampling, grouped read, inherit) on this process and CPU 0;
    /// `inherit` can't be paired with `cpu == -1`, so a concrete CPU is used.
    pub fn supports_inherited_sample_read() -> bool {
        let mut attr: PerfEventAttr = unsafe { mem::zeroed() };
        attr.size = mem::size_of::<PerfEventAttr>() as u32;
        attr.kind = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CPU_CYCLES;
        attr.sample_type =
            PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_READ;
        attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
        attr.sample_period_or_freq = 1_000_000;
        attr.clock_id = libc::CLOCK_MONOTONIC;
        attr.flags =
            PERF_ATTR_FLAG_DISABLED | PERF_ATTR_FLAG_INHERIT | PERF_ATTR_FLAG_USE_CLOCKID;

        let fd = sys_perf_event_open(&attr, 0, 0, -1, PERF_FLAG_FD_CLOEXEC);
        if fd == -1 {
            return false;
        }
        unsafe {
            libc::close(fd);
        }
        true
    }

    pub fn max_sample_rate() -> Option<u64> {
        let data = std::fs::read_to_string("/proc/sys/kernel/perf_event_max_sample_rate").ok()?;
        data.trim().parse::<u64>().ok()
    }

    pub fn build() -> PerfBuilder {
        PerfBuilder {
            pid: 0,
            all_processes: false,
            cpu: None,
            frequency: 0,
            stack_size: 0,
            reg_mask: 0,
            event_source: EventSource::SwCpuClock,
            extra_events: Vec::new(),
            inherit: false,
            start_disabled: false,
            enable_on_exec: false,
            exclude_kernel: true,
            gather_context_switches: false,
        }
    }

    pub fn enable(&mut self) {
        // This only enables the leader, but that is enough for the whole
        // group: siblings are opened enabled and follow the leader's
        // scheduling.
        let result = unsafe { libc::ioctl(self.fd, PERF_EVENT_IOC_ENABLE as _) };

        assert!(result != -1);
    }

    #[inline]
    pub fn are_events_pending(&self) -> bool {
        let head = unsafe { read_head(self.buffer) };
        head != self.position
    }

    #[inline]
    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// The kernel-assigned id and name of each extra event readable from this
    /// event's samples: the leader itself when an extra event aliases it,
    /// plus the sibling events. The per-sample values arrive via
    /// `SampleRecord::read`, matched by id.
    pub fn extra_event_ids(&self) -> impl Iterator<Item = (u64, &str)> {
        self.leader_alias
            .iter()
            .map(|(id, name)| (*id, name.as_str()))
            .chain(
                self.siblings
                    .iter()
                    .map(|sibling| (sibling.id, sibling.name.as_str())),
            )
    }

    #[inline]
    pub fn iter(&mut self) -> EventIter<'_> {
        EventIter::new(self)
    }
}

#[derive(Debug)]
struct EventRefState {
    buffer: *mut u8,
    size: u64,
    pending_commits: BinaryHeap<cmp::Reverse<(u64, u64)>>,
}

impl EventRefState {
    fn new(buffer: *mut u8, size: u64) -> Self {
        EventRefState {
            buffer,
            size,
            pending_commits: BinaryHeap::new(),
        }
    }

    /// Mark the read of [from, to) as complete.
    /// If reads are completed in-order, then this will advance the tail pointer to `to` immediately.
    /// Otherwise, it will remain in the "pending commit" queue, and committed once all previous
    /// reads are also committed.
    fn try_commit(&mut self, from: u64, to: u64) {
        self.pending_commits.push(cmp::Reverse((from, to)));

        let mut position = unsafe { read_tail(self.buffer) };
        while let Some(&cmp::Reverse((from, to))) = self.pending_commits.peek() {
            if from == position {
                unsafe {
                    write_tail(self.buffer, to);
                }
                position = to;
                self.pending_commits.pop();
            } else {
                break;
            }
        }
    }
}

impl Drop for EventRefState {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.buffer as *mut c_void, (self.size + 4096) as _);
        }
    }
}

/// Handle to a single event in the perf ring buffer.
///
/// On Drop, the event will be "consumed" and the read pointer will be advanced.
///
/// If events are dropped out of order, then it will be added to a list of pending commits and
/// committed when all prior events are also dropped. For this reason, events should be dropped
/// in-order to achieve the lowest overhead.
#[derive(Clone)]
pub struct EventRef {
    buffer: *mut u8,
    buffer_size: usize,
    state: Rc<RefCell<EventRefState>>,
    event_location: RawRecordLocation,
    prev_position: u64,
    position: u64,
    parse_info: RecordParseInfo,
}

impl fmt::Debug for EventRef {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_map()
            .entry(&"location", &self.event_location)
            .entry(&"prev_position", &self.prev_position)
            .entry(&"position", &self.position)
            .finish()
    }
}

impl Drop for EventRef {
    #[inline]
    fn drop(&mut self) {
        self.state
            .borrow_mut()
            .try_commit(self.prev_position, self.position);
    }
}

impl EventRef {
    pub fn get(&self) -> RawEventRecord<'_> {
        let buffer = unsafe { slice::from_raw_parts(self.buffer.offset(4096), self.buffer_size) };

        self.event_location.get(buffer, self.parse_info)
    }
}

pub struct EventIter<'a> {
    perf: &'a mut Perf,
}

impl<'a> EventIter<'a> {
    #[inline]
    fn new(perf: &'a mut Perf) -> Self {
        EventIter { perf }
    }
}

impl Iterator for EventIter<'_> {
    type Item = EventRef;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let perf = &mut self.perf;
        let prev_position = perf.position;
        let event_location = next_raw_event(perf.buffer, perf.size, &mut perf.position)?;
        Some(EventRef {
            buffer: perf.buffer,
            buffer_size: perf.size as usize,
            state: perf.event_ref_state.clone(),
            event_location,
            prev_position,
            position: perf.position,
            parse_info: self.perf.parse_info,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_event_specs_parse() {
        let event: ExtraEvent = "instructions:0:0x1".parse().unwrap();
        assert_eq!(event.event_type, PERF_TYPE_HARDWARE);
        assert_eq!(event.config, 1);
        assert_eq!(event.name, "instructions");

        let event: ExtraEvent = "mem_load_retired.l1_miss:4:0x08d1".parse().unwrap();
        assert_eq!(event.event_type, 4);
        assert_eq!(event.config, 0x08d1);
        assert_eq!(event.name, "mem_load_retired.l1_miss");

        let event: ExtraEvent = "l1d_cache:4:4".parse().unwrap();
        assert_eq!((event.event_type, event.config), (4, 4));
    }

    #[test]
    fn malformed_specs_are_rejected() {
        for spec in [
            "",
            "name-only",
            "name:0",
            ":0:0x1",
            "name:nope:0x1",
            "name:0:nope",
        ] {
            assert!(spec.parse::<ExtraEvent>().is_err(), "{spec:?} should fail");
        }
    }
}
