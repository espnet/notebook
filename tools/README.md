# tools

What keeps this repository honest, rather than anything a reader runs.

| | |
|---|---|
| [`run_notebook.py`](run_notebook.py) | Execute a notebook outside Colab: skips the install cells, since the packages are already there, and replaces `wget` and `tar` with their Python equivalents. This is what each notebook's workflow runs, and what "it was run before it was committed" means. |
| [`check_layout.py`](check_layout.py) | The rule [`Demos/README.md`](../Demos/README.md) states, in a form that fails: the naming, the flatness, that a demo pins a release instead of installing espnet from git, and that it has a workflow and a badge of its own. Runs in pre-commit. |

Both are meant to be run from the repository root:

```sh
python tools/check_layout.py
python tools/run_notebook.py Demos/tts_demo.ipynb
```

## What CI installs: the install cell is the contract

Each notebook's workflow does not list its dependencies. It asks the notebook:

```yaml
python -m pip install $(python tools/run_notebook.py "$NOTEBOOK" --print-install)
```

so a demo that starts needing another extra says so in the cell a reader runs,
and the weekly run installs the same thing without being told twice. That makes
the install cell an interface, and this is what it supports.

### Read as a dependency

| | |
|---|---|
| `!pip install espnet` | at the start of a line, no indentation |
| `%pip install espnet` | the magic and the bang are the same thing here |
| `!pip install -q espnet` | flags pass through to pip as they are |
| `!pip install "espnet[enh]==202610.post1"` | quoting is undone, because the shell would otherwise hand pip a name with quotation marks in it |
| `!pip install espnet  # a comment` | `#` and the rest of the line go |
| `!pip install espnet \`<br>`    librosa` | a command continued over lines is one command |
| several install lines, in one cell or many | all of them, in order |

### Not read

| | why |
|---|---|
| `git+https://github.com/espnet/espnet` | **deliberate.** A demo pins a release — `check_layout.py` fails one that does not — so that it does the same thing in April as it does today, and so that a break on `master` the night before someone opens it is not their problem. Installing espnet from git in CI would also test something other than what a reader gets, which would make a pass here mean nothing. |
| an indented install, inside an `if` | what it asks for depends on a choice made when the notebook runs. The course notebook installs S3PRL only for the WavLM front-ends; CI cannot know which branch a reader takes. |
| `!apt-get install …`, `!cd repo && pip install .` | removed before the notebook is executed, because the runner is not Colab, but not treated as a Python dependency |
| anything in a markdown cell | it is prose |

### Cloned and installed

A tool that is on nobody's index is fetched in two shell lines:

```
!git clone https://github.com/wavlab-speech/versa
!cd versa && pip install .
```

`--print-git-install` returns those as `git+<url>`, and the workflow installs
them with `--no-build-isolation` — ParallelWaveGAN's `setup.py` imports `pip`,
which an isolated build environment does not have. `pip install -e .` counts
the same, and a clone nobody installs is data rather than a dependency.

espnet is never returned this way, whichever form it is written in: a demo pins
a release, and a course notebook installing espnet from git is a bug to fix
rather than a dependency to honour.

`--print-install` prints exactly what it found, so the way to see what CI will
install is to ask it:

```sh
$ python tools/run_notebook.py Demos/enh_demo.ipynb --print-install
-q espnet[enh]==202610.post1 espnet_model_zoo librosa
```

### If a demo needs something this cannot express

It probably belongs in [`../Courses/`](../Courses) rather than in `Demos/`. A
demo is a page that runs top to bottom on a free CPU runner from a pinned
release; a notebook that needs a conditional dependency, a package built from
source, or a model from a private repository is a different kind of document,
and the weekly badge would be promising something it cannot keep.

[`tests/test_run_notebook.py`](../tests/test_run_notebook.py) holds each of the
rows above as a case. Three of them were bugs when it was first written.
