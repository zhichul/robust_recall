#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import timedelta
from typing import Optional, List, Set


def _parse_slurm_time(ts: str) -> Optional[timedelta]:
    """
    Convert a Slurm time string into a `timedelta`.

    Valid Slurm formats
        DD-HH:MM:SS   (e.g.  2-03:04:05)
        HH:MM:SS      (e.g.     03:04:05)
        MM:SS         (e.g.        04:05)

    Returns `None` for N/A, UNLIMITED, INFINITE, or "-".
    """
    ts = ts.strip().upper()
    if ts in {"N/A", "UNLIMITED", "INFINITE", "-"}:
        return None

    days = 0
    if "-" in ts:
        days_part, ts = ts.split("-", 1)
        days = int(days_part)

    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    elif len(parts) == 1:
        hours, minutes, seconds = parts[0], 0, 0
    else:
        raise ValueError(f"Unrecognised Slurm time string: {ts}")

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def running_nodes(job_name: str,
                  user: Optional[str] = None,
                  min_remaining: timedelta = timedelta(hours=1)
                  ) -> List[str]:
    """
    Return the NodeList(s) of all *running* Slurm jobs that

    ▸ belong to `user`          (defaults to $USER)
    ▸ have JobName  == `job_name`
    ▸ have *more than* `min_remaining` wall-clock time left
      (default: 1 hour remaining)

    If several jobs match, every distinct NodeList is returned once.
    """
    if user is None:
        user = os.environ["USER"]

    # squeue format:
    #   %i  JobID        | %j  JobName
    #   %N  NodeList     | %M  Elapsed
    #   %l  TimeLimit
    fmt = "%i|%j|%N|%M|%l"

    cmd = [
        "squeue",
        "-u", user,
        "-n", job_name,
        "-t", "R",        # running only
        "-h",             # no header
        "-o", fmt,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result, file=sys.stderr)

    keep: Set[str] = set()
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        job_id, job_name, nodelist, elapsed_s, limit_s = line.split("|")

        elapsed = _parse_slurm_time(elapsed_s)
        limit = _parse_slurm_time(limit_s)

        # If no time-limit, treat as unlimited → always keep.
        if limit is None or elapsed is None:
            keep.add(nodelist)
            continue

        remaining = limit - elapsed
        if remaining > min_remaining:
            keep.add(nodelist)

    return sorted(keep)   # or list(keep) if ordering is irrelevant


if __name__ == "__main__":
    # Example usage:
    for nl in running_nodes("vllm"):
        print(nl)