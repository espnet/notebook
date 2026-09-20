# tools

What keeps this repository honest, rather than anything a reader runs.

| | |
|---|---|
| [`run_notebook.py`](run_notebook.py) | Execute a notebook outside Colab: skips the install cells, since the packages are already there, and replaces `wget` and `tar` with their Python equivalents. This is what `.github/workflows/run_notebooks.yml` runs, and what "it was run before it was committed" means. |
| [`check_layout.py`](check_layout.py) | The rule [`Demos/README.md`](../Demos/README.md) states, in a form that fails: the naming, the flatness, and that a demo pins a release instead of installing espnet from git. Runs in pre-commit. |

Both are meant to be run from the repository root:

```sh
python tools/check_layout.py
python tools/run_notebook.py Demos/tts_demo.ipynb
```
