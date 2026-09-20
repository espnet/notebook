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
TASKS = ("asr", "s2t", "tts", "enh", "spk", "st", "codec", "slu", "diar", "sds")
# <task>_demo.ipynb, or <task>_<variant>_demo.ipynb when a task has a second
# angle worth its own page - asr_streaming_demo.ipynb against asr_demo.ipynb.
# The variant is not a licence to multiply: every demo has to be listed in
# Demos/README.md with a line saying what it does, which is checked below, so
# a second one cannot appear without someone saying how it differs.
NAME = re.compile(rf"^({'|'.join(TASKS)})(_[a-z0-9]+)?_demo\.ipynb$")


def dead_links(readme, root):
    """Notebook links in a README that point at nothing.

    Flattening Demos/ broke every link in the root README at once and
    nothing said so: a markdown link to a moved file is a 404 on GitHub and
    "Notebook not found" in Colab, neither of which reaches anybody here.
    """
    text = readme.read_text(encoding="utf-8")
    dead = []
    for target in re.findall(r"\]\(([^)]+\.ipynb)\)", text):
        if target.startswith(("http://", "https://")):
            continue
        if not (root / target).is_file():
            dead.append(f"{readme.name}: links to {target}, which is not there")
    return dead


def problems():
    found = []
    root = DEMOS.parent
    found += dead_links(root / "README.md", root)
    found += dead_links(DEMOS / "README.md", DEMOS)
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
    workflows = DEMOS.parent / ".github" / "workflows"
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
        # The badge is the claim that this notebook is run every week. One
        # workflow per notebook, because a badge is per workflow - so the
        # claim is true exactly when that workflow exists and names it.
        workflow = workflows / f"{path.stem}.yml"
        if not workflow.is_file():
            found.append(
                f"{path.name}: has no .github/workflows/{workflow.name}. A demo "
                f"that nothing runs is the thing this directory is for not "
                f"having, and the badge would be a lie"
            )
        elif f"notebook: {path.relative_to(DEMOS.parent)}" not in workflow.read_text(
            encoding="utf-8"
        ):
            # the input it passes, not a mention of the path: the file also
            # names its notebook in a comment and in `paths:`, and either
            # would satisfy a looser check while the workflow ran another one
            found.append(
                f"{workflow.name}: does not name {path.name}. The badge on it "
                f"is what the README shows beside that notebook"
            )
        if f"workflows/{path.stem}.yml/badge.svg" not in text:
            found.append(
                f"{path.name}: carries no badge for its own workflow. It is "
                f"what tells a reader in Colab that this page is run rather "
                f"than hoped for"
            )
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
