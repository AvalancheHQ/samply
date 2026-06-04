//! Decide whether an exec will drop into a SIP-protected (un-injectable)
//! process, and if so warn the user. We never modify the exec — we only observe
//! and print.
//!
//! The warning explains *why* the process can't be profiled (SIP strips the
//! `DYLD_*` variable samply relies on) and what the user can do about it, and is
//! emitted as a GitHub Actions `::warning::` annotation when running in CI so it
//! surfaces in the workflow log.

use core::ffi::{c_void, CStr};
use core::fmt::Write;
use libc::{c_char, c_int};

/// Called from the exec/posix_spawn interposers. If SIP is active and the
/// effective binary (the target itself, or — for a `#!` script — its shebang
/// interpreter) is SIP-protected, print a warning. Always returns; the caller
/// then performs the real, unmodified exec.
pub(super) unsafe fn check_and_warn(path: *const c_char) {
    // If SIP is off, DYLD_* survives even for protected binaries, so the preload
    // still loads and there's nothing to warn about.
    if path.is_null() || !sip_enabled() {
        return;
    }
    let path = CStr::from_ptr(path);

    if is_sip_protected(path) {
        warn(path);
        return;
    }

    // Not a protected binary itself — but it may be a script whose shebang
    // interpreter is one (e.g. `pnpm` -> `#!/usr/bin/env node`). The kernel
    // execs that interpreter, and that's where DYLD_* gets stripped.
    let mut buf = [0u8; libc::PATH_MAX as usize];
    if let Some(interp) = shebang_interpreter(path, &mut buf) {
        if is_sip_protected(interp) {
            warn(interp);
        }
    }
}

/// Is System Integrity Protection enabled?
fn sip_enabled() -> bool {
    // `csr_check(mask)` returns 0 when the active configuration *allows* the
    // operation in `mask` (i.e. that protection is off); non-zero means the
    // protection is active. If unrestricted filesystem access is not allowed,
    // SIP is on.
    const CSR_ALLOW_UNRESTRICTED_FS: u32 = 1 << 1;
    extern "C" {
        fn csr_check(mask: u32) -> c_int;
    }
    unsafe { csr_check(CSR_ALLOW_UNRESTRICTED_FS) != 0 }
}

/// Is `path` a SIP-protected system binary? Such binaries carry the restricted
/// file flag, which is what makes the kernel strip `DYLD_*` on exec.
unsafe fn is_sip_protected(path: &CStr) -> bool {
    const SF_RESTRICTED: u32 = 0x0008_0000; // from <sys/stat.h>
    let mut st: libc::stat = core::mem::zeroed();
    if libc::stat(path.as_ptr(), &mut st) != 0 {
        return false;
    }
    st.st_flags & SF_RESTRICTED != 0
}

const SHEBANG_READ_SZ: usize = 256;

/// If `path` is a regular file with a `#!` shebang, return the interpreter path
/// (written into `buf`). Only the interpreter is needed — we don't rebuild argv.
unsafe fn shebang_interpreter<'a>(path: &CStr, buf: &'a mut [u8]) -> Option<&'a CStr> {
    // Only regular files can be scripts (also resolves relative paths via CWD).
    let mut st: libc::stat = core::mem::zeroed();
    if libc::stat(path.as_ptr(), &mut st) != 0 || (st.st_mode & libc::S_IFMT) != libc::S_IFREG {
        return None;
    }

    // Read the first bytes looking for a shebang. `libc::open` here is NOT routed
    // back through our own interpose hook: dyld does not re-interpose an image's
    // references to symbols it itself interposes.
    let mut read_buf = [0u8; SHEBANG_READ_SZ];
    let fd = libc::open(path.as_ptr(), libc::O_RDONLY);
    if fd < 0 {
        return None;
    }
    let nread = libc::read(fd, read_buf.as_mut_ptr() as *mut c_void, read_buf.len());
    libc::close(fd);
    if nread < 2 || read_buf[0] != b'#' || read_buf[1] != b'!' {
        return None;
    }

    // Parse "#!  <interpreter> [<arg>] \n" — we only want <interpreter>.
    let line = match read_buf[..nread as usize].iter().position(|&b| b == b'\n') {
        Some(pos) => &read_buf[..pos],
        None => return None, // truncated / no newline within the window
    };

    let mut start = 2;
    while start < line.len() && (line[start] == b' ' || line[start] == b'\t') {
        start += 1;
    }
    if start >= line.len() {
        return None;
    }
    let interp_end = line[start..]
        .iter()
        .position(|&b| b == b' ' || b == b'\t')
        .map(|p| start + p)
        .unwrap_or(line.len());

    let interp = &line[start..interp_end];
    if interp.is_empty() || interp.len() >= buf.len() {
        return None;
    }
    buf[..interp.len()].copy_from_slice(interp);
    buf[interp.len()] = 0;
    CStr::from_bytes_with_nul(&buf[..=interp.len()]).ok()
}

/// Print the warning to stderr, as a GitHub Actions annotation when in CI.
unsafe fn warn(binary: &CStr) {
    let bin = core::str::from_utf8(binary.to_bytes()).unwrap_or("<unknown>");

    let mut msg = heapless::String::<1024>::new();
    let prefix = if in_github_actions() {
        "::warning title=CodSpeed cannot profile a system process::"
    } else {
        "[CodSpeed] warning: "
    };

    const SIP_DOCS_URL: &str = "https://codspeed.io/docs/instruments/walltime/macos-profiling";

    let _ = writeln!(
        msg,
        "{prefix}CodSpeed could not profile the system process `{bin}`. System Integrity Protection (SIP) removes the DYLD_INSERT_LIBRARIES \
         environment variable that samply uses to attach to a process whenever a protected Apple system binary is executed, so this process and its children \
         are invisible to the profiler. See {SIP_DOCS_URL} for more information."
    );

    libc::write(
        libc::STDERR_FILENO,
        msg.as_ptr() as *const c_void,
        msg.len(),
    );
}

/// True iff `GITHUB_ACTIONS=true` in the environment.
unsafe fn in_github_actions() -> bool {
    let v = libc::getenv(c"GITHUB_ACTIONS".as_ptr());
    !v.is_null() && CStr::from_ptr(v).to_bytes() == b"true"
}
