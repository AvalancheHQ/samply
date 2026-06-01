//! Characterization test: `samply record` cannot profile a workload launched
//! through `/usr/bin/env`.
//!
//! `/usr/bin/env` is an Apple *platform binary*. When dyld execs a platform
//! binary under SIP it strips every `DYLD_*` variable from the environment, so
//! the `DYLD_INSERT_LIBRARIES=<preload>` that samply injects for descendant
//! profiling is gone the instant the kernel resolves a `#!/usr/bin/env bash`
//! shebang. Everything below that first `env` exec — bash, and the `python3`
//! it runs — therefore never loads `samply-mac-preload`, never volunteers its
//! mach task port, and never appears in the profile.
//!
//! This test launches, under the built `samply` binary, a shell script with a
//! `#!/usr/bin/env bash` shebang that runs a short CPU-bound python program, and
//! asserts:
//!   1. the python program actually ran (marker file) — so the negative result
//!      below is not vacuous, and
//!   2. inside python the `DYLD_INSERT_LIBRARIES` var is empty — i.e. it really
//!      was stripped by the `env` exec, and
//!   3. python does NOT appear in the profile (and in fact, because the very
//!      root samply launched is `/usr/bin/env`, samply can't obtain even the
//!      root task and writes no profile at all — which trivially satisfies "no
//!      python profiled").
//!
//! This is a stable characterization of the platform-binary limitation: the
//! `BASH_ENV` re-injection mechanism does NOT change this exact case (a bare
//! `env`-shebang script as samply's direct target still can't be profiled,
//! since `env` is the root). BASH_ENV helps the *descendants* of a profilable
//! root re-acquire `DYLD_INSERT_LIBRARIES` after a shell strips it; it cannot
//! resurrect a root that was itself a platform binary.
//!
//! macOS-only (`cfg(target_os = "macos")`). Launch-mode profiling needs no
//! `task_for_pid` entitlements, so no `samply setup` is required.
//!
//! Run with:
//!   cargo test -p samply --test env_strips_dyld -- --nocapture
#![cfg(target_os = "macos")]

use std::io::Read;
use std::path::Path;
use std::process::Command;

/// Resolve `python3` from PATH, or return None so the test can skip cleanly on
/// the rare runner without it.
fn find_python3() -> Option<String> {
    let out = Command::new("/usr/bin/which").arg("python3").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// Decompress `profile.json.gz` and return the parsed JSON.
fn read_profile(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).expect("profile not written");
    let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .expect("profile is not valid gzip");
    serde_json::from_str(&json).expect("profile is not valid JSON")
}

/// Names (thread `name` + `processName`) of every thread in the profile.
fn thread_names(profile: &serde_json::Value) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(threads) = profile.get("threads").and_then(|t| t.as_array()) {
        for t in threads {
            for key in ["name", "processName"] {
                if let Some(s) = t.get(key).and_then(|v| v.as_str()) {
                    names.push(s.to_string());
                }
            }
        }
    }
    names
}

#[test]
fn python_under_usr_bin_env_is_not_profiled() {
    let Some(python) = find_python3() else {
        eprintln!("skipping: python3 not found on PATH");
        return;
    };
    let samply = env!("CARGO_BIN_EXE_samply");

    let tmp = std::env::temp_dir().join(format!("samply_env_strip_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let marker = tmp.join("marker.txt");
    let profile = tmp.join("profile.json.gz");

    // CPU-bound python: if it were profiled it would produce plenty of samples.
    // It also records its pid and the DYLD var it sees, proving whether the
    // preload's environment survived the `env` exec.
    let pyscript = tmp.join("work.py");
    std::fs::write(
        &pyscript,
        r#"
import os, sys, time
with open(sys.argv[1], "w") as f:
    f.write("pid=%d\n" % os.getpid())
    f.write("dyld=%s\n" % os.environ.get("DYLD_INSERT_LIBRARIES", ""))
    f.write("server=%s\n" % os.environ.get("SAMPLY_BOOTSTRAP_SERVER_NAME", ""))
end = time.time() + 1.5
x = 0
while time.time() < end:
    x = (x + 1) % 1_000_000
print(x)
"#,
    )
    .unwrap();

    // The crucial bit: a `#!/usr/bin/env bash` shebang. The kernel turns running
    // this script into `exec("/usr/bin/env", ["bash", script, ...])`, and that
    // first platform-binary exec is where DYLD_* gets stripped.
    let script = tmp.join("run.sh");
    std::fs::write(
        &script,
        "#!/usr/bin/env bash\nexec \"$1\" \"$2\" \"$3\"\n",
    )
    .unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // samply record --save-only -o <profile> -- <script> <python> <work.py> <marker>
    //
    // We do NOT assert samply succeeds: because the root it launches is
    // `/usr/bin/env` (a platform binary), samply typically fails with
    // "Could not obtain the root task" and writes no profile. That failure is
    // itself the symptom under test, so we just record it for context.
    let output = Command::new(samply)
        .args(["record", "--save-only", "-o"])
        .arg(&profile)
        .arg("--")
        .arg(&script)
        .arg(&python)
        .arg(&pyscript)
        .arg(&marker)
        .output()
        .expect("failed to run samply");
    eprintln!(
        "samply exited {:?}\n--- samply stderr ---\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    // (1) The workload actually ran.
    let marker_contents = std::fs::read_to_string(&marker)
        .expect("python program did not run (no marker) — test would be vacuous");

    // (2) DYLD_INSERT_LIBRARIES was stripped by the `/usr/bin/env` exec: the
    //     `dyld=` line is present but empty.
    let dyld_line = marker_contents
        .lines()
        .find(|l| l.starts_with("dyld="))
        .expect("marker missing dyld= line");
    assert_eq!(
        dyld_line, "dyld=",
        "expected DYLD_INSERT_LIBRARIES to be stripped by /usr/bin/env, but python saw: {dyld_line:?}"
    );

    // (3) No python process/thread made it into the profile. If samply wrote no
    //     profile at all (the common case here — the `env` root has no task
    //     port), that trivially satisfies the assertion.
    let (names, profiled_python): (Vec<String>, Vec<String>) = if profile.exists() {
        let parsed = read_profile(&profile);
        let names = thread_names(&parsed);
        let python = names
            .iter()
            .filter(|n| n.to_lowercase().contains("python"))
            .cloned()
            .collect();
        (names, python)
    } else {
        eprintln!("no profile written (samply could not profile the env root)");
        (Vec::new(), Vec::new())
    };

    // Clean up before asserting (so a failure still leaves a clean tree).
    let _ = std::fs::remove_dir_all(&tmp);

    assert!(
        profiled_python.is_empty(),
        "expected python NOT to be profiled (it runs below /usr/bin/env, which \
         strips DYLD_INSERT_LIBRARIES), but found these threads in the profile: \
         {profiled_python:?}. All thread names: {names:?}"
    );
}
