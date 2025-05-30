import os
import subprocess
import sys


def running_nodes(job_name: str, user: str | None = None) -> list[str]:
    """
    Return the NodeList(s) of all *running* Slurm jobs that
    - belong to `user`          (defaults to the current `$USER`)
    - have JobName == `job_name`

    If several jobs match, every distinct NodeList is returned once.
    """
    if user is None:
        user = os.environ["USER"]

    # squeue options:
    #   -u <user>           filter by user
    #   -n <name>           filter by job name
    #   -h                  no header line
    #   -t R                running jobs only  (drop if you want all states)
    #   -o "<fmt>"          custom output: JobID|JobName|NodeList
    fmt = "%i|%j|%N"        # fields are pipe-separated to simplify splitting

    cmd = [
        "squeue",
        "-u", user,
        "-n", job_name,
        "-t", "R",
        "-h",
        "-o", fmt,
    ]
    # `text=True` -> str instead of bytes; `check=True` raises on non-zero exit
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result, file=sys.stderr)
    nodelists = {
        line.split("|")[2]
        for line in result.stdout.strip().splitlines()
        if line               # ignore the blank line when nothing matches
    }
    return sorted(nodelists)  # or list(nodelists) if order doesn’t matter
