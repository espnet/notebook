# Demos

[![Checked weekly](https://github.com/espnet/notebook/actions/workflows/run_notebooks.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/run_notebooks.yml)

One notebook per task, each short enough to read in a sitting and run before
you lose interest. Open one in Colab, run it top to bottom, hear or see the
result.

The badge is every notebook in the table below, executed cell by cell every
Sunday against the release they pin. It is one badge for the set, so red means
one of them broke rather than all of them — but it is earned: nothing goes in
that table until it runs here. `unmaintained/` has no badge for the same
reason.

| Notebook | What it does |
|---|---|
| [`asr_demo.ipynb`](asr_demo.ipynb) | Transcribe speech with OWSM-CTC, and let it work out the language |
| [`asr_streaming_demo.ipynb`](asr_streaming_demo.ipynb) | Watch the words appear while the audio is still arriving |
| [`st_demo.ipynb`](st_demo.ipynb) | Translate English speech into German, French and Chinese — the same model |
| [`tts_demo.ipynb`](tts_demo.ipynb) | Type a sentence, hear it spoken — one English voice, then 128 of them |
| [`enh_demo.ipynb`](enh_demo.ipynb) | Pull speech out of noise, and measure how much it helped |
| [`spk_demo.ipynb`](spk_demo.ipynb) | Turn a voice into a vector, and score two recordings against each other |
| [`codec_demo.ipynb`](codec_demo.ipynb) | Compress a waveform to a few integers a frame, rebuild it, count the bits |
| [`sds_demo.ipynb`](sds_demo.ipynb) | Speech in, speech out, with a language model thinking in between |

## The naming rule

`<task>_demo.ipynb` or `<task>_<variant>_demo.ipynb`, flat in this directory,
where `<task>` is the ESPnet task
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

A task may have a second angle worth its own page, and then the name carries
it: `asr_streaming_demo.ipynb` beside `asr_demo.ipynb`. That is not a licence
to multiply — this directory filled up once with notebooks nobody could tell
apart. Every demo has to appear in the table above with a line saying what it
does, and `../tools/check_layout.py` fails if one does not, so a second page
cannot arrive without someone saying how it differs from the first.

## What a demo owes the reader

- **It runs, and goes on running.** Every notebook here was executed end to end,
  on the release it pins, before it was committed — and again every Sunday by
  `.github/workflows/run_notebooks.yml`, which is what the badge at the top
  reports. A demo that does not run is worse than no demo: it costs the reader
  their afternoon and the project their credibility.
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
