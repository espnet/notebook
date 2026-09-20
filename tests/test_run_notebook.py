"""What `--print-install` and `drop_installs` make of a notebook's cells.

`--print-install` is what CI installs: the workflow asks the notebook rather
than being told twice. That makes this parsing the single source of truth for
every dependency the weekly run has, so it is worth holding to cases rather
than to a reading of the regex.
"""

import importlib.util
import pathlib
import types

import pytest

TOOL = pathlib.Path(__file__).parents[1] / "tools" / "run_notebook.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_notebook", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook(*sources):
    """A stand-in for nbformat's object: cells with .cell_type and .source."""
    return types.SimpleNamespace(
        cells=[
            types.SimpleNamespace(cell_type="code", source=s) for s in sources
        ]
    )


def install(*sources):
    return _module().pip_arguments(notebook(*sources))


# --- what CI installs -------------------------------------------------------


def test_a_plain_install():
    assert install("!pip install espnet==202610.post1") == ["espnet==202610.post1"]


def test_the_magic_and_the_bang_are_the_same_thing():
    assert install("%pip install librosa") == install("!pip install librosa")


def test_quoting_is_undone_because_a_shell_will_not_do_it_again():
    # pip install $(...) would otherwise get a package name with quotes in it
    assert install('%pip install -q "espnet[enh]==202610.post1"') == [
        "-q",
        "espnet[enh]==202610.post1",
    ]


def test_every_install_cell_counts():
    assert install("!pip install a", "x = 1", "!pip install b c") == ["a", "b", "c"]


def test_several_installs_in_one_cell():
    assert install("!pip install a\nprint(1)\n!pip install b") == ["a", "b"]


def test_a_conditional_install_is_not_ours_to_make():
    # what an indented install asks for depends on a choice made when the
    # notebook runs; the model chooser installs S3PRL only for WavLM
    source = 'if feature == "wavlm":\n  !pip install s3prl'
    assert install(source) == []


def test_a_git_install_is_left_out():
    # those are the notebooks not yet pinned; installing espnet from git here
    # would test something other than what a reader gets
    assert install("!pip install git+https://github.com/espnet/espnet") == []


def test_a_comment_is_not_a_package():
    assert install("!pip install espnet  # the release, not master") == ["espnet"]


def test_a_command_split_over_lines():
    source = "!pip install espnet \\\n    librosa \\\n    soundfile"
    assert install(source) == ["espnet", "librosa", "soundfile"]


def test_markdown_is_not_read():
    module = _module()
    cells = notebook("!pip install real").cells
    cells.append(types.SimpleNamespace(cell_type="markdown", source="!pip install fake"))
    assert module.pip_arguments(types.SimpleNamespace(cells=cells)) == ["real"]


def test_a_notebook_that_installs_nothing():
    assert install("import espnet", "print(1)") == []


# --- what the runner executes ----------------------------------------------


def test_the_install_lines_go_and_the_rest_stays():
    source = "import os\n!pip install espnet\nprint(os.getcwd())"
    assert _module().drop_installs(source) == "import os\nprint(os.getcwd())"


def test_an_indented_install_leaves_its_if_a_body():
    source = 'if feature == "wavlm":\n  !pip install s3prl'
    kept = _module().drop_installs(source)
    assert kept == 'if feature == "wavlm":\n  pass'
    compile(kept, "<test>", "exec")  # the point: it is still Python


def test_a_multiline_install_goes_entirely():
    source = "x = 1\n!pip install espnet \\\n    librosa\ny = 2"
    kept = _module().drop_installs(source)
    assert "librosa" not in kept
    compile(kept, "<test>", "exec")


@pytest.mark.parametrize(
    "line",
    ["!apt-get install -y sox", "!apt install sox", "!cd repo && pip install ."],
)
def test_the_other_installs_go_too(line):
    assert _module().drop_installs(f"x = 1\n{line}\ny = 2") == "x = 1\ny = 2"
