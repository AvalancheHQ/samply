use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, unbounded};
use fxprof_processed_profile::Profile;

use super::error::SamplingError;
use super::process_launcher::{
    get_all_descendant_pids, task_for_pid_and_queue, ExistingProcessRunner, MachError,
    ReceivedStuff, RootTaskRunner, TaskAccepter, TaskLauncher,
};
use super::sampler::{ProcessSpecificPath, Sampler, TaskInit, TaskInitOrShutdown};
use super::time::get_monotonic_timestamp;
use crate::shared::prop_types::{
    ProcessLaunchProps, ProfileCreationProps, RecordingMode, RecordingProps,
};

pub fn run(
    recording_mode: RecordingMode,
    recording_props: RecordingProps,
    mut profile_creation_props: ProfileCreationProps,
) -> Result<(Profile, ExitStatus), MachError> {
    let mut task_accepter = TaskAccepter::new()?;

    let (root_pid_tx, root_pid_rx) = bounded::<u32>(1);
    let mut launch_mode = false;

    let mut root_task_runner: Box<dyn RootTaskRunner> = match recording_mode {
        RecordingMode::All => {
            eprintln!("Error: Profiling all processes is not supported on macOS.");
            eprintln!("You can only profile processes which you launch via samply, or attach to via --pid.");
            std::process::exit(1)
        }
        RecordingMode::Pid(pid) => Box::new(ExistingProcessRunner::new(pid, &mut task_accepter)),
        RecordingMode::Launch(process_launch_props) => {
            launch_mode = true;
            let ProcessLaunchProps {
                mut env_vars,
                command_name,
                args,
                iteration_count,
                ignore_exit_code,
            } = process_launch_props;

            let task_launcher = if profile_creation_props.coreclr.any_enabled() {
                // We need to set DOTNET_PerfMapEnabled=3 in the environment if it's not already set.
                // If we set it, we'll also set unlink_aux_files=true to avoid leaving files
                // behind in the temp directory. But if it's set manually, assume the user
                // knows what they're doing and will specify the arg as needed.
                if !env_vars.iter().any(|p| p.0 == "DOTNET_PerfMapEnabled") {
                    env_vars.push(("DOTNET_PerfMapEnabled".into(), "3".into()));
                    profile_creation_props.unlink_aux_files = true;
                }

                // To be filled in with new launching code in future PR

                TaskLauncher::new(
                    &command_name,
                    &args,
                    iteration_count,
                    ignore_exit_code,
                    &env_vars,
                    task_accepter.extra_env_vars(),
                )?
            } else {
                TaskLauncher::new(
                    &command_name,
                    &args,
                    iteration_count,
                    ignore_exit_code,
                    &env_vars,
                    task_accepter.extra_env_vars(),
                )?
            };

            let mut task_launcher = task_launcher;
            task_launcher.set_root_pid_sender(root_pid_tx);
            Box::new(task_launcher)
        }
    };

    let (task_sender, task_receiver) = unbounded();

    let sampler_thread = thread::spawn(move || {
        let sampler = Sampler::new(task_receiver, recording_props, profile_creation_props);
        sampler.run()
    });

    // Shared "pids we've already registered a task for". The IPC accepter loop and the
    // descendant poller both consult this to avoid registering the same pid twice when
    // the dylib injection and the poller race (the dylib essentially always wins, but we
    // still need to be defensive).
    let seen_pids: Arc<Mutex<HashSet<u32>>> = Arc::new(Mutex::new(HashSet::new()));

    // Descendant poller: catches processes whose ancestors stripped DYLD_INSERT_LIBRARIES
    // (hardened binaries like /usr/bin/env, node, Electron). Only runs in launch mode.
    // Polls proc_listpids recursively under the root pid; new pids get task_for_pid'd
    // and pushed onto the accepter queue, mirroring --pid attach semantics. Stops after
    // 5 seconds — long enough to catch the early hardened-exec chain but not so long
    // that we burn CPU for the whole profile.
    let (poll_stop_tx, poll_stop_rx) = bounded::<()>(1);
    let poll_thread = if launch_mode {
        let queue_handle = task_accepter.queue_handle();
        let seen_pids = Arc::clone(&seen_pids);
        let poll_interval_ms: u64 = std::env::var("SAMPLY_DESCENDANT_POLL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        Some(thread::spawn(move || {
            let Ok(root_pid) = root_pid_rx.recv_timeout(Duration::from_secs(5)) else {
                return;
            };
            let deadline = Instant::now() + Duration::from_secs(5);
            let interval = Duration::from_millis(poll_interval_ms);
            while Instant::now() < deadline {
                if poll_stop_rx.try_recv().is_ok() {
                    break;
                }
                for pid in get_all_descendant_pids(root_pid) {
                    // Atomically claim the pid before doing task_for_pid; if the IPC path
                    // already grabbed it, skip.
                    if !seen_pids.lock().unwrap().insert(pid) {
                        continue;
                    }
                    if !task_for_pid_and_queue(pid, &queue_handle, true) {
                        // task_for_pid failed (process already exited, or missing
                        // entitlements). Roll back the claim so another path could
                        // retry. In practice this means leaving the pid out of the
                        // tree this poll cycle.
                        seen_pids.lock().unwrap().remove(&pid);
                    }
                }
                thread::sleep(interval);
            }
        }))
    } else {
        drop(root_pid_rx);
        None
    };

    let (accepter_sender, accepter_receiver) = unbounded();
    let accepter_thread_seen_pids = Arc::clone(&seen_pids);
    let accepter_thread = thread::spawn(move || {
        let seen_pids = accepter_thread_seen_pids;
        // Loop while accepting messages from the spawned process tree.

        // A map of pids to channel senders, to notify existing tasks of Jitdump
        // paths. Having the mapping here lets us deliver the path to the right
        // task even in cases where a process execs into a new task with the same pid.
        let mut path_senders_per_pid = HashMap::new();

        loop {
            if let Ok(()) = accepter_receiver.try_recv() {
                task_sender.send(TaskInitOrShutdown::Shutdown).ok();
                break;
            }
            let timeout = Duration::from_secs_f64(1.0);
            match task_accepter.next_message(timeout) {
                Ok(ReceivedStuff::AcceptedTask(accepted_task)) => {
                    let pid = accepted_task.get_id();
                    let newly_seen = seen_pids.lock().unwrap().insert(pid);
                    if !newly_seen {
                        // The poller already registered this pid. Don't push a duplicate
                        // task into the sampler, but still unblock the dylib-injected
                        // child so it can run.
                        accepted_task.start_execution();
                        continue;
                    }
                    let (path_sender, path_receiver) = unbounded();
                    let send_result = task_sender.send(TaskInitOrShutdown::TaskInit(TaskInit {
                        start_time_mono: get_monotonic_timestamp(),
                        task: accepted_task.task(),
                        pid,
                        path_receiver,
                    }));
                    path_senders_per_pid.insert(pid, path_sender);
                    if send_result.is_err() {
                        // The sampler has already shut down. This task arrived too late.
                    }
                    accepted_task.start_execution();
                }
                Ok(ReceivedStuff::JitdumpPath(pid, path)) => {
                    match path_senders_per_pid.entry(pid) {
                        Entry::Occupied(mut entry) => {
                            let send_result =
                                entry.get_mut().send(ProcessSpecificPath::Jitdump(path));
                            if send_result.is_err() {
                                // The task is probably already dead. The path arrived too late.
                                entry.remove();
                            }
                        }
                        Entry::Vacant(_entry) => {
                            eprintln!(
                                "Received a Jitdump path for pid {pid} which I don't have a task for."
                            );
                        }
                    }
                }
                Ok(ReceivedStuff::MarkerFilePath(pid, path)) => {
                    match path_senders_per_pid.entry(pid) {
                        Entry::Occupied(mut entry) => {
                            let send_result =
                                entry.get_mut().send(ProcessSpecificPath::MarkerFile(path));
                            if send_result.is_err() {
                                // The task is probably already dead. The path arrived too late.
                                entry.remove();
                            }
                        }
                        Entry::Vacant(_entry) => {
                            eprintln!(
                                "Received a marker file path for pid {pid} which I don't have a task for."
                            );
                        }
                    }
                }
                Ok(ReceivedStuff::DotnetTracePath(pid, path)) => {
                    match path_senders_per_pid.entry(pid) {
                        Entry::Occupied(mut entry) => {
                            let send_result =
                                entry.get_mut().send(ProcessSpecificPath::DotnetTrace(path));
                            if send_result.is_err() {
                                // The task is probably already dead. The path arrived too late.
                                entry.remove();
                            }
                        }
                        Entry::Vacant(_entry) => {
                            eprintln!(
                                "Received a marker file path for pid {pid} which I don't have a task for."
                            );
                        }
                    }
                }
                Err(MachError::RcvTimedOut) => {
                    // TODO: give status back via task_sender
                }
                Err(err) => {
                    eprintln!("Encountered error while waiting for task port: {err:?}");
                }
            }
        }
    });

    // Run the root task: either launch or attach to existing pid
    let exit_status = root_task_runner.run_root_task()?;

    // Stop the descendant poller before the accepter, so that any pids the poller
    // pushed are drained by the accepter loop before it shuts down.
    let _ = poll_stop_tx.send(());
    if let Some(t) = poll_thread {
        let _ = t.join();
    }

    accepter_sender
        .send(())
        .expect("couldn't tell accepter thread to stop");
    accepter_thread
        .join()
        .expect("couldn't join accepter thread");

    // Wait for the sampler to stop. It will run until all accepted tasks have terminated,
    // or until the time limit has elapsed.
    let profile_result = sampler_thread.join().expect("couldn't join sampler thread");

    let profile = match profile_result {
        Ok(profile) => profile,
        Err(SamplingError::CouldNotObtainRootTask) => {
            eprintln!("Profiling failed: Could not obtain the root task.");
            eprintln!();
            eprintln!("On macOS, samply cannot profile system commands (sleep, system python, hardened binaries) because they strip DYLD_INSERT_LIBRARIES on exec.");
            eprintln!("samply also attempted to attach to descendant processes via task_for_pid, but none became available within 5 seconds.");
            eprintln!();
            eprintln!("Suggested remedy: Run 'samply setup' to grant task_for_pid entitlements, or profile a binary you compiled yourself (cargo install, Homebrew, etc.).");
            std::process::exit(1)
        }
        Err(e) => {
            eprintln!("An error occurred during profiling: {e}");
            std::process::exit(1)
        }
    };

    Ok((profile, exit_status))
}
