//! Thin libc helpers (no_std): env lookup, argv counting, file copy, and the
//! ad-hoc codesign shell-out.

use core::{ffi::CStr, ptr};
use libc::{c_char, c_int, c_void, pid_t};

extern "C" {
    static environ: *const *const c_char;
}

pub(super) unsafe fn getenv(var: &CStr) -> Option<&'static CStr> {
    let value = libc::getenv(var.as_ptr());
    if value.is_null() {
        return None;
    }
    Some(CStr::from_ptr(value))
}

pub(super) unsafe fn count_argv(argv: *const *const c_char) -> usize {
    if argv.is_null() {
        return 0;
    }
    let mut n = 0;
    while !(*argv.add(n)).is_null() {
        n += 1;
    }
    n
}

struct FileDescriptor(c_int);

impl FileDescriptor {
    unsafe fn open(path: &CStr, flags: c_int, mode: libc::c_uint) -> Option<Self> {
        let fd = libc::open(path.as_ptr(), flags, mode);
        (fd >= 0).then_some(Self(fd))
    }

    fn as_raw(&self) -> c_int {
        self.0
    }
}

impl Drop for FileDescriptor {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.0);
        }
    }
}

pub(super) unsafe fn copy_file(src: &CStr, dst: &CStr) -> bool {
    let Some(in_fd) = FileDescriptor::open(src, libc::O_RDONLY, 0) else {
        return false;
    };
    let Some(out_fd) =
        FileDescriptor::open(dst, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC, 0o755)
    else {
        return false;
    };

    let mut buf = [0u8; 1 << 16];
    loop {
        let n = libc::read(in_fd.as_raw(), buf.as_mut_ptr() as *mut c_void, buf.len());
        if n < 0 {
            return false;
        }
        if n == 0 {
            return true;
        }
        if !write_all(out_fd.as_raw(), &buf[..n as usize]) {
            return false;
        }
    }
}

unsafe fn write_all(fd: c_int, mut buf: &[u8]) -> bool {
    while !buf.is_empty() {
        let n = libc::write(fd, buf.as_ptr() as *const c_void, buf.len());
        if n <= 0 {
            return false;
        }
        buf = &buf[n as usize..];
    }
    true
}

/// Run `/usr/bin/codesign -s - -f <file>` synchronously, output to /dev/null.
/// Calling `libc::posix_spawn` here reaches the *real* spawn: dyld does not
/// interpose an image's references to symbols it itself interposes, so this does
/// not recurse through `samply_sip_posix_spawn`.
pub(super) unsafe fn codesign_adhoc(file: &CStr) -> bool {
    let argv: [*const c_char; 6] = [
        c"/usr/bin/codesign".as_ptr(),
        c"-s".as_ptr(),
        c"-".as_ptr(),
        c"-f".as_ptr(),
        file.as_ptr(),
        ptr::null(),
    ];

    let mut fa: libc::posix_spawn_file_actions_t = core::mem::zeroed();
    if libc::posix_spawn_file_actions_init(&mut fa) != 0 {
        return false;
    }
    let devnull = c"/dev/null".as_ptr();
    libc::posix_spawn_file_actions_addopen(&mut fa, 1, devnull, libc::O_WRONLY, 0);
    libc::posix_spawn_file_actions_addopen(&mut fa, 2, devnull, libc::O_WRONLY, 0);

    let mut pid: pid_t = 0;
    let rc = libc::posix_spawn(
        &mut pid,
        argv[0],
        &fa,
        ptr::null(),
        argv.as_ptr() as *const *mut c_char,
        environ as *const *mut c_char,
    );
    libc::posix_spawn_file_actions_destroy(&mut fa);
    if rc != 0 {
        return false;
    }

    let mut status: c_int = 0;
    loop {
        let r = libc::waitpid(pid, &mut status, 0);
        if r >= 0 {
            break;
        }
        if *libc::__error() != libc::EINTR {
            return false;
        }
    }
    libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0
}
