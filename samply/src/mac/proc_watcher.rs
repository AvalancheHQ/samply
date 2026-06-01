//! Launch-mode descendant watcher.
//!
//! samply normally profiles a launched tree by injecting `DYLD_INSERT_LIBRARIES`
//! so each descendant loads `samply-mac-preload` and volunteers its mach task
//! port over IPC. That breaks the moment the tree crosses a process that strips
//! `DYLD_*` — most importantly `/usr/bin/env`, an Apple *platform binary*. A
//! command like `samply record -- pnpm …` resolves through `#!/usr/bin/env node`,
//! so the very first exec drops the preload and nothing below it ever connects.
//! samply then has no task at all and aborts with "Could not obtain the root
//! task", discarding a perfectly good run.
//!
//! This watcher is the fallback: it polls the descendant tree of the launched
//! root pid and, for every process it is allowed to inspect, grabs the task port
//! directly with `task_for_pid` and hands it to the sampler — no preload needed.
//! Node (and locally-built binaries) ship `get-task-allow`, so they become
//! profilable this way even though they never loaded the preload. It requires
//! samply itself to hold the `com.apple.security.cs.debugger` entitlement (run
//! `samply setup`); without it `task_for_pid` fails and the watcher simply
//! attaches nothing.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};

use super::process_launcher::{get_all_descendant_pids, task_for_pid_checked};
use super::sampler::{TaskInit, TaskInitOrShutdown};
use super::time::get_monotonic_timestamp;

/// Poll the descendant tree of the launched root pid, attaching to every process
/// we can via `task_for_pid` and forwarding it to the sampler.
///
/// `seen_pids` is shared with the IPC accepter loop so each pid is forwarded to
/// the sampler at most once, no matter which path reaches it first. Blocks until
/// `stop_rx` fires (sent once the root process tree has exited).
pub fn watch_descendants(
    root_pid_rx: Receiver<u32>,
    task_sender: Sender<TaskInitOrShutdown>,
    seen_pids: Arc<Mutex<HashSet<u32>>>,
    stop_rx: Receiver<()>,
) {
    let Ok(root_pid) = root_pid_rx.recv_timeout(Duration::from_secs(5)) else {
        log::debug!("proc_watcher: timed out waiting for the root pid");
        return;
    };
    log::debug!("proc_watcher: watching descendants of root pid {root_pid}");

    let poll_interval = Duration::from_millis(
        std::env::var("SAMPLY_DESCENDANT_POLL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50),
    );

    let mut roots = vec![root_pid];
    // Pids seen in the previous poll. We only attach a pid once it has been
    // visible for a full interval, which gives the faster preload-IPC path first
    // claim on any process that *will* load the preload — so we don't steal a
    // preload process out from under its jitdump/marker-path stream (which only
    // the IPC accepter can route). For trees that never load the preload (the
    // `/usr/bin/env` case) the pid simply persists and is attached next round.
    let mut prev_round: HashSet<u32> = HashSet::new();
    loop {
        // Pick up root pids reported by later `--iteration-count` launches.
        while let Ok(extra_root) = root_pid_rx.try_recv() {
            roots.push(extra_root);
        }

        let mut this_round: HashSet<u32> = HashSet::new();
        for &root in &roots {
            this_round.insert(root);
            this_round.extend(get_all_descendant_pids(root));
        }

        for &pid in &this_round {
            if prev_round.contains(&pid) {
                attempt_attach(pid, &task_sender, &seen_pids);
            }
        }
        prev_round = this_round;

        // Sleep until the next poll, but wake immediately when asked to stop.
        if stop_rx.recv_timeout(poll_interval).is_ok() {
            log::debug!("proc_watcher: stopping");
            break;
        }
    }
}

/// `task_for_pid` `pid` and forward it to the sampler, unless another path has
/// already claimed it.
fn attempt_attach(
    pid: u32,
    task_sender: &Sender<TaskInitOrShutdown>,
    seen_pids: &Arc<Mutex<HashSet<u32>>>,
) {
    // Claim the pid first so we never double-register with the IPC path.
    if !seen_pids.lock().unwrap().insert(pid) {
        return;
    }

    let Some(task) = task_for_pid_checked(pid) else {
        // Roll the claim back: the preload-IPC path might still reach this pid
        // even though task_for_pid couldn't (e.g. it's about to load the preload).
        seen_pids.lock().unwrap().remove(&pid);
        return;
    };

    log::debug!("proc_watcher: attached pid {pid} via task_for_pid");

    // These processes never loaded the preload, so there is no jitdump/marker
    // path stream for them. Hand the sampler a receiver whose sender we drop
    // immediately; `check_received_paths` then sees an empty, closed channel.
    let (_path_sender, path_receiver) = crossbeam_channel::unbounded();
    let send_result = task_sender.send(TaskInitOrShutdown::TaskInit(TaskInit {
        start_time_mono: get_monotonic_timestamp(),
        task,
        pid,
        path_receiver,
    }));
    if send_result.is_err() {
        // The sampler has already shut down.
        seen_pids.lock().unwrap().remove(&pid);
    }
}
