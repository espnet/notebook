# Demos

One notebook per task, each short enough to read in a sitting and run before
you lose interest. Open one in Colab, run it top to bottom, hear or see the
result.

| Notebook | What it does |
|---|---|
| [`tts_demo.ipynb`](tts_demo.ipynb) | Type a sentence, hear it spoken — one English voice, then 128 of them |

## The naming rule

`<task>_demo.ipynb`, flat in this directory, where `<task>` is the ESPnet task
name the rest of the toolkit already uses:

| `<task>` | |
|---|---|
| `asr` | speech recognition |
| `s2t` | OWSM: recognition, translation and language ID in one model |
| `tts` | text-to-speech |
| `enh` | speech enhancement and separation |
| `spk` | speaker embedding, verification, diarization |
| `st` | speech translation |
| `codec` | neural audio codecs |

These are the words `espnet.load(model, task=...)`, `espnet2/tasks/` and the
model cards use, so a reader who knows one knows the others. Flat, because a
directory per task held one or two files and made you click twice to find out
there was nothing there.

One notebook per task, not one per idea: a second `asr_something_demo.ipynb`
is how this directory filled up with notebooks nobody could tell apart. If a
task needs more than one page, the second one belongs in `../Courses/` with the
material that explains it.

## What a demo owes the reader

- **It runs.** Every notebook here was executed end to end, on the release it
  pins, before it was committed. A demo that does not run is worse than no demo:
  it costs the reader their afternoon and the project their credibility.
- **It pins a release.** `espnet==<version>`, never `git+https://...`. The
  notebook should do the same thing in April that it does today, and a break on
  `master` the night before someone opens it is not their problem.
- **It ends by pointing onwards** — the one-line terminal command, the Space if
  there is one, the course notebook for the long version.

## `unmaintained/`

Everything that was here before, in one place. Most of it does not run: the
oldest is from 2021 and installs packages that no longer resolve. They are kept
because some of them are still the only written record of how a thing was done,
and removed as they are replaced.

**Nothing in there is checked.** Do not send someone to one of those notebooks
without opening it yourself first.
