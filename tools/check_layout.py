#!/usr/bin/env python3
"""Hold Demos/ to the rule its README states.

A convention written only in prose is one nobody notices breaking. This is
the same rule, in a form that fails.

    python tools/check_layout.py
"""

import pathlib
import re
import sys

DEMOS = pathlib.Path(__file__).parents[1] / "Demos"
UNMAINTAINED = DEMOS / "unmaintained"
# the task names espnet2/tasks/ and espnet.load(task=...) use
TASKS = ("asr", "s2t", "tts", "enh", "spk", "st", "codec", "slu", "diar")
# <task>_demo.ipynb, or <task>_<variant>_demo.ipynb when a task has a second
# angle worth its own page - asr_streaming_demo.ipynb against asr_demo.ipynb.
# The variant is not a licence to multiply: every demo has to be listed in
# Demos/README.md with a line saying what it does, which is checked below, so
# a second one cannot appear without someone saying how it differs.
NAME = re.compile(rf"^({'|'.join(TASKS)})(_[a-z0-9]+)?_demo\.ipynb$")


def problems():
    found = []
    for path in sorted(DEMOS.glob("*.ipynb")):
        if not NAME.match(path.name):
            found.append(
                f"{path.name}: not <task>_demo.ipynb - the tasks are "
                f"{', '.join(TASKS)}. A notebook that is not one task's demo "
                f"belongs in ../Courses/ or in unmaintained/"
            )
    for path in sorted(DEMOS.rglob("*.ipynb")):
        if path.parent not in (DEMOS, UNMAINTAINED):
            found.append(
                f"{path.relative_to(DEMOS)}: Demos/ is flat - a demo goes here, "
                f"an old one in unmaintained/, and everything else in ../Courses/"
            )
    index = (DEMOS / "README.md").read_text(encoding="utf-8")
    for path in sorted(DEMOS.glob("*.ipynb")):
        if path.name not in index:
            found.append(
                f"{path.name}: not listed in Demos/README.md. A demo nobody "
                f"can tell apart from the others is how this directory filled "
                f"up before; say in one line what this one does"
            )
    for path in sorted(DEMOS.glob("*.ipynb")):
        text = path.read_text(encoding="utf-8")
        if "git+https://github.com/espnet/espnet" in text:
            found.append(
                f"{path.name}: installs espnet from git. A demo pins a release, "
                f"so that it does the same thing in April as it does today"
            )
        # espnet, or espnet[extra, extra], pinned to a release. Written out
        # rather than as "anything but a quote": that spelling excluded the
        # letter n, so espnet[enh] failed a check the notebook passed.
        if not re.search(r"espnet(\[[a-z, ]+\])?==\d{6}", text):
            found.append(f"{path.name}: does not pin an espnet release")
    return found


def main():
    found = problems()
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        sys.exit(f"\n{len(found)} problem(s) in Demos/")
    print(f"Demos/ ok: {len(list(DEMOS.glob('*.ipynb')))} demo(s), "
          f"{len(list(UNMAINTAINED.glob('*.ipynb')))} unmaintained")


if __name__ == "__main__":
    main()
