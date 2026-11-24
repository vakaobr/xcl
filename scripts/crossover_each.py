#!/usr/bin/env python3
# Part of xcl (GPL-2.0-only).
"""Run every test of a libtest binary in its own CrossOver/Wine process.

Each test runs as: <exe> --exact <name> --test-threads=1 --nocapture

GPU tests (name contains 'gpu_') get --gpu-cap seconds (default 60),
the rest get --cap seconds (default 180). After a timeout the bottle's
wineserver is killed so no process is left holding the next run.

Status per test: pass / fail / ignored / skipped / timeout / crash.
A 'skipped' test printed a SKIPPED line from the skip_test helper and
still passed according to libtest.

Writes results incrementally to a JSON file and prints a summary at the
end with counts per status.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

DEFAULT_CX = "/Applications/CrossOver.app/Contents/SharedSupport/CrossOver"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run each test of a libtest binary in its own CrossOver process.",
    )
    p.add_argument("exe", help="Path to the libtest binary (.exe)")
    p.add_argument("output", help="Path to write the JSON results file")
    p.add_argument(
        "--bottle", required=True,
        help="CrossOver bottle name (e.g. xcl-test)",
    )
    p.add_argument(
        "--cx-path", default=DEFAULT_CX,
        help="CrossOver SharedSupport path (default: %(default)s)",
    )
    p.add_argument(
        "--gpu-cap", type=float, default=60,
        help="Timeout in seconds for GPU tests (default: %(default)s)",
    )
    p.add_argument(
        "--cap", type=float, default=180,
        help="Timeout in seconds for non-GPU tests (default: %(default)s)",
    )
    p.add_argument(
        "--filter",
        help="Only run tests whose name contains this substring",
    )
    p.add_argument(
        "--dxc-dir",
        help="Wine path to the DXC directory (sets XCL_DXC_DIR, e.g. Z:\\path\\to\\dxc\\bin\\x64)",
    )
    return p.parse_args()


def bottle_dir(bottle):
    return os.path.expanduser(
        f"~/Library/Application Support/CrossOver/Bottles/{bottle}"
    )


def wine(cx, bottle, args, timeout, extra_env=None):
    """Run a command through CrossOver's wine and return (rc, output, timed_out)."""
    cmd = [f"{cx}/bin/wine", "--bottle", bottle, *args]
    env = None
    if extra_env:
        env = dict(os.environ, **extra_env)
    try:
        p = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, errors="replace", timeout=timeout, env=env,
        )
        return p.returncode, p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as e:
        out = (
            (e.stdout or b"").decode(errors="replace")
            + (e.stderr or b"").decode(errors="replace")
        )
        return None, out, True


def kill_bottle(cx, bottle):
    """Kill only this bottle's wineserver using its WINEPREFIX."""
    prefix = bottle_dir(bottle)
    env = dict(os.environ, WINEPREFIX=prefix)
    subprocess.run(
        [f"{cx}/bin/wineserver", "-k"],
        env=env, capture_output=True, timeout=60,
    )
    time.sleep(1)


def list_tests(cx, bottle, exe, extra_env=None):
    """List all tests in the binary."""
    rc, out, _ = wine(cx, bottle, [exe, "--list", "--format", "terse"], 300, extra_env)
    names = [
        line[: -len(": test")]
        for line in out.splitlines()
        if line.endswith(": test")
    ]
    if not names:
        sys.exit(f"no tests listed from {exe} (exit {rc}):\n{out[-2000:]}")
    return names


def first_reason(out):
    """Extract a short reason string from the test output."""
    keep = [
        line for line in out.splitlines()
        if line.strip()
        and not re.match(r"^[0-9a-f]{4}:(fixme|err|warn|trace):", line)
    ]
    for pat in ("panicked at", "assertion", "left:", "SKIPPED", "error", "Error"):
        for i, line in enumerate(keep):
            if pat in line:
                return " | ".join(s.strip() for s in keep[i : i + 3])[:300]
    return " | ".join(s.strip() for s in keep[-3:])[:300]


def classify(out):
    """Classify a test result from its output."""
    if re.search(r"test result: ok\. 0 passed; 0 failed; 1 ignored", out):
        return "ignored"
    if re.search(r"test result: ok\. 1 passed", out):
        if re.search(r"^SKIPPED", out, re.M):
            return "skipped"
        return "pass"
    if re.search(r"test result: FAILED", out):
        return "fail"
    return "crash"


def main():
    args = parse_args()

    extra_env = {}
    if args.dxc_dir:
        extra_env["XCL_DXC_DIR"] = args.dxc_dir

    names = list_tests(args.cx_path, args.bottle, args.exe, extra_env or None)
    if args.filter:
        names = [n for n in names if args.filter in n]

    if not names:
        sys.exit("no tests matched the filter")

    results = []
    for i, name in enumerate(names, 1):
        limit = args.gpu_cap if "gpu_" in name else args.cap
        start = time.time()
        rc, out, timed_out = wine(
            args.cx_path, args.bottle,
            [args.exe, "--exact", name, "--test-threads=1", "--nocapture"],
            limit, extra_env or None,
        )
        secs = round(time.time() - start, 1)

        if timed_out:
            status = "timeout"
            kill_bottle(args.cx_path, args.bottle)
        else:
            status = classify(out)

        reason = "" if status == "pass" else first_reason(out)
        results.append({
            "name": name, "status": status,
            "secs": secs, "rc": rc, "reason": reason,
        })

        line = f"[{i}/{len(names)}] {status:8} {secs:6}s {name}"
        if reason:
            line += f"  -- {reason[:140]}"
        print(line, flush=True)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)

    # Summary
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total = len(results)

    print()
    print(f"Total: {total}")
    for status in ("pass", "skipped", "ignored", "fail", "timeout", "crash"):
        count = counts.get(status, 0)
        if count:
            print(f"  {status}: {count}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
