//! Interpose `execve` + `posix_spawn{,p}` to detect when a process is about to
//! exec a SIP-protected Apple system binary.
//!
//! macOS strips every `DYLD_*` environment variable when the kernel execs a
//! protected platform binary (`/usr/bin/env`, `/bin/sh`, system `python`, …).
//! Once stripped, this preload no longer loads into that process *or any of its
//! descendants*, so samply stops seeing the whole subtree. We can't keep the
//! preload alive across that boundary here — we only *observe* it and print a
//! warning naming the offending binary (see [`detect`]). The real exec then
//! proceeds completely unmodified.

use libc::{c_char, c_int, pid_t, posix_spawn_file_actions_t, posix_spawnattr_t};

use crate::InterposeEntry;

mod detect;

unsafe extern "C" fn samply_sip_execve(
    path: *const c_char,
    argv: *const *const c_char,
    envp: *const *const c_char,
) -> c_int {
    detect::check_and_warn(path);
    libc::execve(path, argv, envp)
}

unsafe extern "C" fn samply_sip_posix_spawn(
    pid: *mut pid_t,
    path: *const c_char,
    file_actions: *const posix_spawn_file_actions_t,
    attrp: *const posix_spawnattr_t,
    argv: *const *mut c_char,
    envp: *const *mut c_char,
) -> c_int {
    detect::check_and_warn(path);
    libc::posix_spawn(pid, path, file_actions, attrp, argv, envp)
}

unsafe extern "C" fn samply_sip_posix_spawnp(
    pid: *mut pid_t,
    file: *const c_char,
    file_actions: *const posix_spawn_file_actions_t,
    attrp: *const posix_spawnattr_t,
    argv: *const *mut c_char,
    envp: *const *mut c_char,
) -> c_int {
    // `file` may be a bare name (e.g. "sh") that the real call resolves against
    // `$PATH`. No need to resolve it here: libc's `posix_spawnp` funnels into
    // `posix_spawn` with the resolved path, which hits our interposer above.
    detect::check_and_warn(file);
    libc::posix_spawnp(pid, file, file_actions, attrp, argv, envp)
}

// We only interpose `execve` + `posix_spawn{,p}`: the other `exec*` wrappers
// (`execv`, `execvp`, `execl`, …) funnel through `execve` inside libc, and
// `posix_spawn{,p}` are the separate spawn syscalls used by libuv/Node and
// CPython.
#[used]
#[allow(non_upper_case_globals)]
#[link_section = "__DATA,__interpose"]
pub static mut _interpose_execve: InterposeEntry =
    InterposeEntry::new(samply_sip_execve as *const (), libc::execve as *const ());

#[used]
#[allow(non_upper_case_globals)]
#[link_section = "__DATA,__interpose"]
pub static mut _interpose_posix_spawn: InterposeEntry = InterposeEntry::new(
    samply_sip_posix_spawn as *const (),
    libc::posix_spawn as *const (),
);

#[used]
#[allow(non_upper_case_globals)]
#[link_section = "__DATA,__interpose"]
pub static mut _interpose_posix_spawnp: InterposeEntry = InterposeEntry::new(
    samply_sip_posix_spawnp as *const (),
    libc::posix_spawnp as *const (),
);
