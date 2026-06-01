use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossbeam_channel::Receiver;

use super::process_launcher::{get_all_descendant_pids, task_for_pid_and_queue, ReceivedStuff};

/// Register `pid` with a kqueue to deliver NOTE_FORK and NOTE_EXIT events.
/// Errors (e.g. ESRCH when the process has already exited) are silently ignored.
fn watch_pid(kq: libc::c_int, pid: u32) {
    let change = libc::kevent {
        ident: pid as libc::uintptr_t,
        filter: libc::EVFILT_PROC,
        flags: libc::EV_ADD | libc::EV_CLEAR,
        fflags: libc::NOTE_FORK | libc::NOTE_EXIT,
        data: 0,
        udata: std::ptr::null_mut(),
    };
    unsafe {
        libc::kevent(kq, &change, 1, std::ptr::null_mut(), 0, std::ptr::null());
    }
}

fn try_attach(pid: u32, queue: &Arc<Mutex<Vec<ReceivedStuff>>>, already_queued: &mut HashSet<u32>) {
    if already_queued.contains(&pid) {
        return;
    }
    if task_for_pid_and_queue(pid, queue, true) {
        log::debug!("proc_watcher: task_for_pid attached pid={pid}");
        already_queued.insert(pid);
    } else {
        log::debug!("proc_watcher: task_for_pid failed for pid={pid} (will rely on IPC)");
    }
}

fn register_and_attach(
    kq: libc::c_int,
    pid: u32,
    queue: &Arc<Mutex<Vec<ReceivedStuff>>>,
    watched: &mut HashSet<u32>,
    already_queued: &mut HashSet<u32>,
) {
    if !watched.insert(pid) {
        return;
    }
    log::debug!("proc_watcher: watching pid={pid}");
    watch_pid(kq, pid);
    try_attach(pid, queue, already_queued);
}

/// Watch descendants of `root_pid` using kqueue, attaching via task_for_pid when
/// available and relying on the preload IPC path otherwise.
///
/// Blocks until `stop_rx` receives a message. Spawned as a dedicated thread in
/// launch mode by [`crate::mac::profiler::run`].
pub fn watch_descendants(
    root_pid_rx: Receiver<u32>,
    queue_handle: Arc<Mutex<Vec<ReceivedStuff>>>,
    stop_rx: Receiver<()>,
) {
    let Ok(root_pid) = root_pid_rx.recv_timeout(Duration::from_secs(5)) else {
        log::debug!("proc_watcher: timed out waiting for root_pid");
        return;
    };

    log::debug!("proc_watcher: starting, root_pid={root_pid}");

    let kq = unsafe { libc::kqueue() };
    if kq == -1 {
        log::warn!("proc_watcher: kqueue() failed");
        return;
    }

    let mut watched: HashSet<u32> = HashSet::new();
    let mut already_queued: HashSet<u32> = HashSet::new();

    // Seed: watch root_pid plus any descendants already running, closing the
    // race between process start and our kqueue registration.
    for pid in std::iter::once(root_pid).chain(get_all_descendant_pids(root_pid)) {
        register_and_attach(kq, pid, &queue_handle, &mut watched, &mut already_queued);
    }

    loop {
        if stop_rx.try_recv().is_ok() {
            log::debug!("proc_watcher: stopping");
            break;
        }

        // Block up to 100 ms so we notice stop signals promptly while spending
        // essentially zero CPU between fork events.
        let timeout = libc::timespec {
            tv_sec: 0,
            tv_nsec: 100_000_000,
        };
        let mut events = [unsafe { std::mem::zeroed::<libc::kevent>() }; 32];
        let n = unsafe {
            libc::kevent(
                kq,
                std::ptr::null(),
                0,
                events.as_mut_ptr(),
                events.len() as libc::c_int,
                &timeout,
            )
        };
        if n < 0 {
            break;
        }

        for event in events.iter().take(n as usize) {
            let fflags = event.fflags;
            let pid = event.ident as u32;

            if fflags & libc::NOTE_FORK != 0 {
                // macOS NOTE_FORK does not deliver the child pid in event.data
                // (always 0, unlike FreeBSD). Scan from the forking process to
                // discover its new children and any grandchildren already running.
                log::debug!("proc_watcher: NOTE_FORK on pid={pid}, scanning descendants");
                for new_pid in get_all_descendant_pids(pid) {
                    register_and_attach(
                        kq,
                        new_pid,
                        &queue_handle,
                        &mut watched,
                        &mut already_queued,
                    );
                }
            }

            if fflags & libc::NOTE_EXIT != 0 {
                log::debug!("proc_watcher: NOTE_EXIT pid={pid}");
                watched.remove(&pid);
            }
        }
    }

    unsafe { libc::close(kq) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// kqueue NOTE_FORK fires when the watched process spawns a child.
    /// Also confirms that on macOS event.data is 0 (child pid not delivered).
    #[test]
    fn note_fork_fires_on_child_spawn() {
        let kq = unsafe { libc::kqueue() };
        assert!(kq != -1, "kqueue() failed");

        watch_pid(kq, std::process::id());

        let mut child = std::process::Command::new("/usr/bin/true")
            .spawn()
            .expect("failed to spawn /usr/bin/true");
        child.wait().unwrap();

        let timeout = libc::timespec {
            tv_sec: 1,
            tv_nsec: 0,
        };
        let mut events = [unsafe { std::mem::zeroed::<libc::kevent>() }; 8];
        let n = unsafe {
            libc::kevent(kq, std::ptr::null(), 0, events.as_mut_ptr(), 8, &timeout)
        };

        assert!(n > 0, "expected at least one kqueue event, got {n}");
        let fork_event = events
            .iter()
            .take(n as usize)
            .find(|e| e.fflags & libc::NOTE_FORK != 0)
            .expect("expected NOTE_FORK event");

        // macOS does not deliver the child pid in data (always 0).
        // Copy to a local to avoid packed-struct alignment issues.
        let data = fork_event.data;
        assert_eq!(data, 0, "macOS NOTE_FORK should have data=0, got {data}");

        unsafe { libc::close(kq) };
    }

    /// BASH_ENV is sourced by bash for non-interactive scripts, so a value
    /// exported there is visible to the script — and crucially survives the
    /// hardened /usr/bin/env exec chain that strips DYLD_* variables.
    fn run_script_with_bash_env(shebang: &str) -> String {
        let tmpdir = tempfile::tempdir().unwrap();

        let env_file = tmpdir.path().join("env.sh");
        std::fs::write(&env_file, "export SAMPLY_TEST_VAR=injected\n").unwrap();

        let script = tmpdir.path().join("script.sh");
        std::fs::write(
            &script,
            format!("{shebang}\necho $SAMPLY_TEST_VAR\n"),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let output = std::process::Command::new(&script)
            .env("BASH_ENV", &env_file)
            .env_remove("SAMPLY_TEST_VAR")
            .output()
            .expect("failed to run script");

        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    #[test]
    fn bash_env_injects_through_usr_bin_env_bash() {
        assert_eq!(
            run_script_with_bash_env("#!/usr/bin/env bash"),
            "injected"
        );
    }

    #[test]
    fn bash_env_injects_through_bin_bash() {
        assert_eq!(run_script_with_bash_env("#!/bin/bash"), "injected");
    }

    #[test]
    fn bash_env_injects_through_usr_bin_env_bin_bash() {
        assert_eq!(
            run_script_with_bash_env("#!/usr/bin/env /bin/bash"),
            "injected"
        );
    }
}
