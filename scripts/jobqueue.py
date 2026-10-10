#!/usr/bin/env python3
"""Shared job queue for the local GPUs (standard library only). One worker per GPU (scripts/queue_worker.sh)
pops the first eligible job; this module does the bookkeeping under a file lock.

Queue file $QDIR/queue.txt (default runs/queue), one job per line, '#' starts a comment line:
    <name> <outdir> <res> <needs> <module> <args...>
  res    std | smol | phi -- host-memory class. smol (SmolLM3-3B) and phi (Phi-4-mini) train with the optimizer
         offloaded to host memory and run one per class at a time (two Phi offload jobs do not fit in host RAM);
         a running process of the same model outside the queue (e.g. started by hand) also counts.
  needs  '-' or a run json that must be complete (contain a test block) first, e.g. the stage-1 run whose
         checkpoint a fork loads.
The worker runs  python -m openjev_rlcd.<module> <args...> --output <outdir>/<name>.json.
A job whose <outdir>/<name>.json is already complete is dropped; a failed job is re-queued once, at the end.

    jobqueue.py pop --gpu 0 --worker PID   print the popped job line (exit 0); exit 3 = all blocked, 4 = queue empty
    jobqueue.py finish --gpu 0 --name N --out D --ok 0|1 --rc RC
    jobqueue.py show
    jobqueue.py push [--front] < job_lines
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from pathlib import Path

QDIR = Path(os.environ.get("QDIR", "runs/queue"))
# model-name substring that identifies a running process of each offload class (overridable for tests)
RES_MODEL = dict(kv.split("=") for kv in os.environ.get("QUEUE_RES_MODELS", "phi=Phi-4-mini,smol=SmolLM3").split(","))
MIN_FREE_GB = {"smol": 45, "phi": 55}
# a bash process running one of the experiment scripts (not e.g. "bash -c <command line mentioning a model>")
SCRIPT = re.compile(r"(\S*/)?bash\s+(\S*/)?(run_task|run|backbone|two_stage|baselines|ablations|worker)\.sh(\s|$)")  # MemAvailable needed before an offload job may start
MAX_FAILS = 2


def now():
    return time.strftime("%m-%d %H:%M")


def say(msg):
    with open(QDIR / "status.txt", "a") as f:
        f.write(f"{now()} {msg}\n")


def complete(path):
    try:
        with open(path) as f:
            return '"test": {' in f.read()
    except OSError:
        return False


def is_job(line):
    s = line.strip()
    return bool(s) and not s.startswith("#")


def parse(line):
    f = line.split()
    if len(f) < 5 or f[2] not in ("std", "smol", "phi"):
        raise ValueError(f"bad queue line: {line!r}")
    return dict(name=f[0], out=f[1], res=f[2], needs=f[3], module=f[4], line=line.strip())


def read_lines(path):
    return path.read_text().splitlines() if path.exists() else []


def write_lines(path, lines):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(x + "\n" for x in lines))
    os.replace(tmp, path)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def running_entries():
    """Lines of running.txt: <gpu> <worker pid> <start epoch> <job line>"""
    out = []
    for line in read_lines(QDIR / "running.txt"):
        if line.strip():
            gpu, pid, t0, job = line.split(" ", 3)
            out.append(dict(gpu=gpu, pid=int(pid), t0=t0, job=parse(job), raw=line))
    return out


def processes():
    """(pid, comm, args) of every process (comm and args are both kept: a script's comm is its interpreter)."""
    out = []
    ps = subprocess.run(["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True).stdout.splitlines()
    for line in ps:
        f = line.split(None, 2)
        if len(f) == 3:
            out.append((int(f[0]), f[1], f[2]))
    return out


def external_classes():
    """Offload classes with a live training process outside our bookkeeping: a python process of openjev_rlcd or
    a bash experiment script, mentioning the model (so the gaps between the runs of a script also count)."""
    found = set()
    for pid, comm, args in processes():
        if pid == os.getpid():
            continue
        training = (comm.startswith("python") and "openjev_rlcd" in args) or (comm == "bash" and SCRIPT.match(args))
        if training:
            for res, sub in RES_MODEL.items():
                if sub in args:
                    found.add(res)
    return found


def job_pid(job):
    """pid of a live process running this job (its --output argument), if any."""
    target = f"--output {job['out']}/{job['name']}.json"
    for pid, comm, args in processes():
        if target in args and comm != "bash" and pid != os.getpid():
            return pid
    return None


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


class Lock:
    def __enter__(self):
        QDIR.mkdir(parents=True, exist_ok=True)
        self.f = open(QDIR / ".lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def pop(gpu, worker):
    qpath = QDIR / "queue.txt"
    with Lock():
        lines = read_lines(qpath)
        running = []
        for r in running_entries():  # a dead worker's job goes back to the front of the queue
            if pid_alive(r["pid"]):
                running.append(r)
                continue
            orphan = job_pid(r["job"])
            if orphan is not None:  # the worker died but its run lives on: track the run itself
                r["pid"], r["raw"] = orphan, f"{r['gpu']} {orphan} {r['t0']} {r['job']['line']}"
                running.append(r)
                say(f"gpu{r['gpu']} worker gone, run {r['job']['out']}/{r['job']['name']} still alive (pid {orphan})")
            elif not complete(Path(r["job"]["out"]) / f"{r['job']['name']}.json"):
                say(f"gpu{r['gpu']} worker {r['pid']} gone: re-queue {r['job']['out']}/{r['job']['name']}")
                lines.insert(0, r["job"]["line"])
        busy = {r["job"]["res"] for r in running} | external_classes()
        free_gb = mem_available_gb()
        keep, chosen, seen = [], None, set()
        for line in lines:
            if not is_job(line):
                keep.append(line)
                continue
            j = parse(line)
            key = (j["out"], j["name"])
            if complete(Path(j["out"]) / f"{j['name']}.json"):
                say(f"drop {j['out']}/{j['name']} (already complete)")
                continue
            if key in seen or any((r["job"]["out"], r["job"]["name"]) == key for r in running):
                say(f"drop duplicate {j['out']}/{j['name']}")
                continue
            seen.add(key)
            ok = j["needs"] == "-" or complete(j["needs"])
            if j["res"] in MIN_FREE_GB:
                ok = ok and j["res"] not in busy and free_gb >= MIN_FREE_GB[j["res"]]
            if chosen is None and ok:
                chosen = j
                continue
            keep.append(line)
        write_lines(qpath, keep)
        if chosen is None:
            write_lines(QDIR / "running.txt", [r["raw"] for r in running])
            return 3 if any(is_job(x) for x in keep) else 4
        running_lines = [r["raw"] for r in running] + [f"{gpu} {worker} {int(time.time())} {chosen['line']}"]
        write_lines(QDIR / "running.txt", running_lines)
    print(chosen["line"])
    return 0


def finish(gpu, name, out, ok, rc):
    with Lock():
        mine, others = None, []
        for r in running_entries():
            if mine is None and r["gpu"] == str(gpu) and r["job"]["name"] == name and r["job"]["out"] == out:
                mine = r
            else:
                others.append(r["raw"])
        write_lines(QDIR / "running.txt", others)
        with open(QDIR / "history.txt", "a") as f:
            f.write(f"{now()} gpu{gpu} {out}/{name} rc={rc} ok={ok}\n")
        if ok or mine is None:
            return 0
        fails = sum(1 for x in read_lines(QDIR / "history.txt") if f" {out}/{name} " in x and x.endswith("ok=0"))
        log = Path(out) / f"{name}.log"
        if log.exists():
            os.replace(log, Path(out) / f"{name}.fail{fails}.log")
        if fails < MAX_FAILS:
            lines = read_lines(QDIR / "queue.txt") + [mine["job"]["line"]]
            write_lines(QDIR / "queue.txt", lines)
            say(f"gpu{gpu} FAILED {out}/{name} (rc={rc}, failure {fails}): re-queued at the end")
        else:
            say(f"gpu{gpu} FAILED {out}/{name} (rc={rc}, failure {fails}): dropped")
    return 0


def push(front):
    """add job lines read from stdin (validated; already queued or running jobs are skipped), at the front of the
    queue (after its leading comment lines) or at the end"""
    new = [x.strip() for x in sys.stdin.read().splitlines() if is_job(x)]
    for x in new:
        parse(x)
    with Lock():
        lines = read_lines(QDIR / "queue.txt")
        have = {(parse(x)["out"], parse(x)["name"]) for x in lines if is_job(x)}
        have |= {(r["job"]["out"], r["job"]["name"]) for r in running_entries()}
        add = []
        for x in new:
            j = parse(x)
            if (j["out"], j["name"]) in have:
                say(f"push: {j['out']}/{j['name']} already queued or running, skipped")
                continue
            have.add((j["out"], j["name"]))
            add.append(x)
        if front:
            k = 0
            while k < len(lines) and not is_job(lines[k]):
                k += 1
            lines = lines[:k] + add + lines[k:]
        else:
            lines = lines + add
        write_lines(QDIR / "queue.txt", lines)
        for x in add:
            j = parse(x)
            say(f"push{' (front)' if front else ''}: {j['out']}/{j['name']}")
    print(f"added {len(add)} of {len(new)} jobs")
    return 0


def show():
    with Lock():
        running = running_entries()
        jobs = [parse(x) for x in read_lines(QDIR / "queue.txt") if is_job(x)]
    for r in running:
        mins = (time.time() - int(r["t0"])) / 60
        print(f"RUNNING gpu{r['gpu']} {mins:6.0f} min  {r['job']['out']}/{r['job']['name']}")
    print(f"{len(jobs)} queued")
    for j in jobs:
        dep = "" if j["needs"] == "-" or complete(j["needs"]) else f"  (waits for {j['needs']})"
        print(f"  {j['res']:4s} {j['out']}/{j['name']}{dep}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pop")
    p.add_argument("--gpu", required=True)
    p.add_argument("--worker", type=int, required=True)
    f = sub.add_parser("finish")
    f.add_argument("--gpu", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--ok", type=int, required=True)
    f.add_argument("--rc", type=int, required=True)
    sub.add_parser("show")
    u = sub.add_parser("push")
    u.add_argument("--front", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pop":
        return pop(a.gpu, a.worker)
    if a.cmd == "push":
        return push(a.front)
    if a.cmd == "finish":
        return finish(a.gpu, a.name, a.out, a.ok, a.rc)
    return show()


if __name__ == "__main__":
    sys.exit(main())
