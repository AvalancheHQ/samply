#!/usr/bin/env bash
#
# Integration smoke tests for the samply macOS preload (sip_redirect).
#
# These run the *built* dylib under DYLD_INSERT_LIBRARIES with the SIP-redirect
# feature enabled, and assert externally observable behaviour the unit tests
# can't reach (real exec/posix_spawn, real codesign, real env stripping).
#
# They are the regression net for the two bugs found in June 2026:
#   - the shebang-arg use-after-scope (manifested as `env: <garbage>` / exit 127)
#   - turbo-style env stripping hiding the whole subtree
#
# KEY CONSTRAINT: the interpose only fires in a process that actually loaded the
# preload. Apple platform binaries (/bin/bash, /usr/bin/env, …) STRIP DYLD_* on
# exec, so a shell harness is never itself preloaded and cannot drive the
# exec-path tests. `node` honours DYLD (it ships the entitlement), so we use a
# node parent as the "preloaded driver" for those. Tests requiring node are
# skipped (not failed) when node is unavailable or didn't load the preload.
#
# Usage:
#   ./build.sh                 # build the fat dylib first
#   tests/preload_smoke.sh     # run the tests
#
# Exit non-zero on any failure.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DYLIB="$HERE/../binaries/libsamply_mac_preload.dylib"
CACHE="$(mktemp -d /tmp/samply-preload-test.XXXXXX)"
WORK="$(mktemp -d /tmp/samply-preload-work.XXXXXX)"
FAILED=0

cleanup() { rm -rf "$CACHE" "$WORK"; }
trap cleanup EXIT

if [[ ! -f "$DYLIB" ]]; then
  echo "FATAL: $DYLIB not found — run ./build.sh first" >&2
  exit 2
fi

export DYLD_INSERT_LIBRARIES="$DYLIB"
export SAMPLY_SIP_REDIRECT_DIR="$CACHE"
export SAMPLY_BOOTSTRAP_SERVER_NAME="test.bootstrap.name"

pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; echo "       $2" >&2; FAILED=1; }
skip() { echo "skip - $1"; }

NODE="$(command -v node || true)"

# Confirm node actually loads the preload (honours DYLD). If not, the exec-path
# tests below would silently no-op, so we skip them rather than false-pass.
node_is_preloaded() {
  [[ -n "$NODE" ]] || return 1
  [[ "$("$NODE" -e "process.stdout.write(process.env.DYLD_INSERT_LIBRARIES?'y':'n')" 2>/dev/null)" == "y" ]]
}

# ---------------------------------------------------------------------------
# Test 1: shebang redirect through a SIP interpreter does not corrupt argv.
# Regression for the shebang-arg use-after-scope (was: exit 127, garbage arg).
# A preloaded node parent spawns a `#!/usr/bin/env bash` script; the hook must
# rebuild argv as [<cached env>, "bash", <script>, <args…>] without corruption.
# We assert BOTH that it ran correctly AND that the redirect actually fired
# (the re-signed /usr/bin/env copy lands in the cache).
# ---------------------------------------------------------------------------
test_shebang_arg() {
  if ! node_is_preloaded; then skip "shebang redirect (node not preloaded)"; return; fi
  local script="$WORK/hello.sh"
  printf '#!/usr/bin/env bash\necho SHEBANG_OK_$1\n' > "$script"
  chmod +x "$script"
  local out
  out="$("$NODE" -e "
    const r=require('child_process').spawnSync('$script',['world'],{encoding:'utf8'});
    process.stdout.write((r.stdout||'').trim()+'|rc='+r.status);
  " 2>/dev/null)"
  if [[ "$out" == "SHEBANG_OK_world|rc=0" && -f "$CACHE/usr/bin/env" ]]; then
    pass "shebang redirect preserves argv (#!/usr/bin/env bash)"
  else
    fail "shebang redirect preserves argv" "out='$out' env-cached=$([[ -f "$CACHE/usr/bin/env" ]] && echo y || echo n)"
  fi
}

# ---------------------------------------------------------------------------
# Test 2: direct SIP-binary redirect runs a re-signed copy from the cache.
# A preloaded node parent spawns /bin/bash; the hook redirects to a cached,
# ad-hoc-re-signed copy. Assert the copy exists and the command still works.
# ---------------------------------------------------------------------------
test_sip_redirect_caches() {
  if ! node_is_preloaded; then skip "SIP binary redirect (node not preloaded)"; return; fi
  local out
  out="$("$NODE" -e "
    const r=require('child_process').spawnSync('/bin/bash',['-c','echo BASH_OK'],{encoding:'utf8'});
    process.stdout.write((r.stdout||'').trim());
  " 2>/dev/null)"
  if [[ "$out" == "BASH_OK" && -f "$CACHE/bin/bash" ]]; then
    pass "SIP binary redirected to re-signed cache copy (/bin/bash)"
  else
    fail "SIP binary redirect" "out='$out' cached=$([[ -f "$CACHE/bin/bash" ]] && echo y || echo n)"
  fi
}

# ---------------------------------------------------------------------------
# Test 3: setuid binaries are never redirected (re-signing would break them).
# ---------------------------------------------------------------------------
test_setuid_not_redirected() {
  if ! node_is_preloaded; then skip "setuid policy (node not preloaded)"; return; fi
  # Probe a setuid binary through the hook; it must NOT be copied into the cache.
  "$NODE" -e "try{require('child_process').spawnSync('/usr/bin/login',['-h'],{timeout:500})}catch(e){}" >/dev/null 2>&1
  if [[ -f "$CACHE/usr/bin/login" || -f "$CACHE/usr/bin/sudo" ]]; then
    fail "setuid not redirected" "a setuid binary was copied into the cache"
  else
    pass "setuid binaries are not redirected"
  fi
}

echo "== samply preload smoke tests =="
echo "   dylib: $DYLIB"
echo "   cache: $CACHE"
echo "   node : ${NODE:-<none>}"
test_shebang_arg
test_sip_redirect_caches
test_setuid_not_redirected

if [[ $FAILED -ne 0 ]]; then
  echo "== FAILURES =="
  exit 1
fi
echo "== all passed =="
