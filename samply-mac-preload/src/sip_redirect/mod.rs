//! Keep `DYLD_INSERT_LIBRARIES` (and therefore this preload) alive across
//! environment-stripping `exec`s — both SIP system binaries and env-sanitizing
//! user programs.
//!
//! macOS strips every `DYLD_*` environment variable when the kernel execs an
//! Apple *platform binary* in a SIP-protected path (`/usr/bin/env`, `/bin/sh`,
//! `/bin/bash`, the coreutils, …). Once stripped, this preload no longer loads
//! into that process *or any of its descendants*, so samply stops seeing the
//! whole subtree (the classic `env -> bash -> uv -> python` blind spot).
//!
//! Workaround: in every process that *does* have this preload loaded, interpose
//! the `exec`/`posix_spawn` family. When the target is such a SIP binary, run an
//! ad-hoc *re-signed copy* of it from a writable cache instead — see
//! [`redirect`]. The copy is not a platform binary, so dyld keeps `DYLD_*`, the
//! preload re-arms in the child, and the var continues to propagate down the
//! tree. This complements the root `task_for_pid` watcher (which covers the
//! tree root and respawns).
//!
//! ## Shebang handling
//! - When the exec target is a regular file with a `#!` shebang, and the
//!   interpreter is a SIP binary (`#!/usr/bin/env`, `#!/bin/bash`, etc.), we
//!   redirect the *interpreter* instead and rebuild argv to include the script
//!   path.  This catches `pnpm` → `/usr/bin/env node pnpm.js`, `pip` →
//!   `/usr/bin/env python`, and similar npm/pip shebang chains.
//!
use libc::{c_char, c_int, pid_t, posix_spawn_file_actions_t, posix_spawnattr_t};

use crate::InterposeEntry;

mod buffers;
mod redirect;
mod sys;

use buffers::{CPathBuffer, ShebangArgv};
use redirect::redirect;

unsafe extern "C" fn samply_sip_execve(
    path: *const c_char,
    argv: *const *const c_char,
    envp: *const *const c_char,
) -> c_int {
    let mut path_buf = CPathBuffer::new();
    let mut shebang_argv = ShebangArgv::new();

    let p = redirect(path, argv, &mut path_buf, &mut shebang_argv);
    if let Some(sa) = shebang_argv.as_slice() {
        return libc::execve(p, sa.as_ptr(), envp);
    }
    libc::execve(p, argv, envp)
}

unsafe extern "C" fn samply_sip_posix_spawn(
    pid: *mut pid_t,
    path: *const c_char,
    file_actions: *const posix_spawn_file_actions_t,
    attrp: *const posix_spawnattr_t,
    argv: *const *mut c_char,
    envp: *const *mut c_char,
) -> c_int {
    let mut path_buf = CPathBuffer::new();
    let mut shebang_argv = ShebangArgv::new();

    let p = redirect(
        path,
        argv as *const *const c_char,
        &mut path_buf,
        &mut shebang_argv,
    );
    if let Some(sa) = shebang_argv.as_slice() {
        return libc::posix_spawn(
            pid,
            p,
            file_actions,
            attrp,
            sa.as_ptr() as *const *mut c_char,
            envp,
        );
    }
    libc::posix_spawn(pid, p, file_actions, attrp, argv, envp)
}

unsafe extern "C" fn samply_sip_posix_spawnp(
    pid: *mut pid_t,
    file: *const c_char,
    file_actions: *const posix_spawn_file_actions_t,
    attrp: *const posix_spawnattr_t,
    argv: *const *mut c_char,
    envp: *const *mut c_char,
) -> c_int {
    let mut path_buf = CPathBuffer::new();
    let mut shebang_argv = ShebangArgv::new();

    let p = redirect(
        file,
        argv as *const *const c_char,
        &mut path_buf,
        &mut shebang_argv,
    );
    if let Some(sa) = shebang_argv.as_slice() {
        return libc::posix_spawnp(
            pid,
            p,
            file_actions,
            attrp,
            sa.as_ptr() as *const *mut c_char,
            envp,
        );
    }
    libc::posix_spawnp(pid, p, file_actions, attrp, argv, envp)
}

// We only interpose `execve` + `posix_spawn{,p}`: the other `exec*` wrappers
// (`execv`, `execvp`, `execl`, …) funnel through `execve` inside libc, and
// `posix_spawn{,p}` are the separate spawn syscalls used by libuv/Node and
// CPython. Note we must NOT interpose `execvp`: the real `env` relies on its
// PATH-searching behaviour to resolve the command.
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
