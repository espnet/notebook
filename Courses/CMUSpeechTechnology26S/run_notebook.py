"""Run one of these notebooks outside Colab, to see whether it still works.

Colab is where they are meant to be opened, and a runner is not Colab: the
packages are already installed, and `wget` and `tar` may not exist. Those
cells are replaced with local equivalents and everything else runs unchanged,
so what is checked is the ESPnet code, which is the part that rots.

    python run_notebook.py speaker_verification.ipynb [--timeout 3600]

Prints `RESULT ok <notebook>` or `RESULT failed <notebook>` with the cell that
failed, and exits non-zero on failure. `.github/workflows/run_notebooks.yml`
runs it weekly.
"""

import argparse
import json
import pathlib
import re
import sys
import tempfile

# A `git clone` is often the data, not an install, so only the pip and apt
# lines go - including the "cd repo && pip install ." that follows a clone.
INSTALL_LINE = re.compile(
    r"^\s*[!%]\s*(pip|apt|apt-get)\b|^\s*!\s*cd \S+ && pip"
)


def drop_installs(source: str) -> str:
    """Remove the install lines, keeping whatever else the cell does.

    A cell is often both: the model chooser ends with `!pip install s3prl`,
    and skipping the whole cell would lose the variable it sets.
    """
    return "\n".join(
        line for line in source.split("\n") if not INSTALL_LINE.match(line)
    )


def localise(source: str) -> str:
    """Rewrite the shell lines this machine cannot run."""
    lines = []
    for line in source.split("\n"):
        wget = re.match(r"\s*!\s*wget\s+(\S+)(?:\s+-O\s+(\S+))?.*", line)
        tar = re.match(r"\s*!\s*tar\s+-\w+\s+(\S+)", line)
        unzip = re.match(r"\s*!\s*unzip\s+(?:-\w+\s+)?(\S+)", line)
        if wget:
            url, out = wget.group(1), wget.group(2) or ""
            name = out or url.rsplit("/", 1)[-1]
            # honour --no-check-certificate: openslr's certificate is why the
            # notebook passes it, and urlretrieve verifies by default
            unverified = "--no-check-certificate" in line
            lines.append(
                "import ssl as _s, urllib.request as _u; "
                + (
                    "_c = _s._create_unverified_context(); "
                    if unverified
                    else "_c = _s.create_default_context(); "
                )
                + f'open("{name}", "wb").write('
                + f'_u.urlopen("{url}", context=_c).read())'
            )
        elif tar:
            # filter="data" refuses absolute and ../ members: these archives
            # come off the network, one of them over an unverified connection
            lines.append(
                f'import tarfile as _t; '
                f'_t.open("{tar.group(1)}").extractall(".", filter="data")'
            )
        elif unzip:
            lines.append(
                f'import zipfile as _z; _z.ZipFile("{unzip.group(1)}").extractall(".")'
            )
        else:
            lines.append(line)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(args.notebook, as_version=4)
    kept = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            kept.append(cell)
            continue
        source = drop_installs(cell.source)
        if not [line for line in source.split("\n") if line.strip() and not line.strip().startswith("#")]:
            print(f"skipping an install cell: {cell.source.splitlines()[0][:60]}")
            continue
        cell.source = localise(source)
        kept.append(cell)
    # `!command` runs in a shell whose PATH does not include this
    # interpreter's bin directory, so gdown and friends would not be found
    setup = nbformat.v4.new_code_cell(
        "import os, sys\n"
        "os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']"
    )
    nb.cells = [setup] + kept

    workdir = args.workdir or tempfile.mkdtemp(prefix="nbrun-")
    print(f"running {len(nb.cells)} cells in {workdir}")
    client = NotebookClient(
        nb, timeout=args.timeout, kernel_name=args.kernel, resources={"metadata": {"path": workdir}}
    )
    try:
        client.execute()
    except CellExecutionError as e:
        print(f"RESULT failed {args.notebook}")
        message = str(e)
        # the cell source comes first and the traceback last: show both ends
        print(message[:600])
        print("   ...")
        print(message[-1200:])
        return 1
    print(f"RESULT ok {args.notebook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
