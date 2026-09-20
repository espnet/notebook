# CMU Speech Technology 26S

CMU 11492/11692/18495, *Speech Technology for Conversational AI*, Spring 2026.

The demonstration notebooks from the course, kept here so that they keep
working: the graded exercises are removed, the installs point at released
ESPnet rather than at anyone's fork, and each one is run before it is changed.

They are meant to be opened in Colab, and they run on CPU.

The course's two fine-tuning notebooks are not here yet: they are written
against an ESPnet3 data API that has since changed, and both stop at
`recipe_dir must be set when data_src is None`. They come back when they are
updated to the current one.

| Notebook | | What it does |
|---|---|---|
| [`speaker_verification.ipynb`](speaker_verification.ipynb) | [![speaker_verification](https://github.com/espnet/notebook/actions/workflows/speaker_verification.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speaker_verification.yml) | Speaker embeddings with ESPnet-SPK, verification, and a simple diarization |
| [`speech_enhancement.ipynb`](speech_enhancement.ipynb) | [![speech_enhancement](https://github.com/espnet/notebook/actions/workflows/speech_enhancement.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_enhancement.yml) | Enhancement and separation, scored with VERSA and a pretrained ASR model |
| [`text_to_speech.ipynb`](text_to_speech.ipynb) | [![text_to_speech](https://github.com/espnet/notebook/actions/workflows/text_to_speech.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/text_to_speech.yml) | Single-speaker and multi-speaker synthesis, and VERSA scores |
| [`neural_codec.ipynb`](neural_codec.ipynb) | [![neural_codec](https://github.com/espnet/notebook/actions/workflows/neural_codec.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/neural_codec.yml) | Three pretrained neural codecs and the bitrate trade between them |
| [`speech_translation.ipynb`](speech_translation.ipynb) | [![speech_translation](https://github.com/espnet/notebook/actions/workflows/speech_translation.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_translation.yml) | Offline and simultaneous speech translation with ESPnet-ST-v2 |

All five run every Sunday, each on its own workflow, cell by cell, against the
release the notebook pins. A badge is that notebook and nothing else.

## Samples the course used and this repository cannot carry

Two of the recordings in `speech_enhancement` came from Google Drive copies of
licensed corpora. Neither could be committed here, so each was replaced by
something the notebook can fetch on its own:

- **CHiME-4.** The real noisy sample is now `ped.wav` from the challenge's own
  [data page](https://www.chimechallenge.org/challenges/chime4/data), which
  publishes a few recordings to listen to. The corpus itself is LDC2017S24 and
  built on WSJ0. The page's sample is single-channel where the corpus has six;
  the enhancement in the notebook is single-channel either way, so the
  demonstration is unchanged.
- **wsj0-2mix.** The separation model is trained on mixtures built from WSJ0,
  which cannot be published, and the course's mixture came from Drive. The
  notebook now builds one the same way at runtime — two ESPnet test recordings
  of different speakers, summed at equal level and normalised. It is a
  different mixture from the course's, and an easier one: two clean utterances
  with no shared channel. The separation and the ASR scoring after it still
  show what they were there to show.

## Running them outside Colab

`../../tools/run_notebook.py` executes one of them here: it skips the install
cells, since the packages are already present, and replaces `wget`, `tar` and
`unzip` with their Python equivalents.

```sh
python ../../tools/run_notebook.py speaker_verification.ipynb --print-install
```

prints what that notebook installs, so an environment for it is

```sh
pip install $(python ../../tools/run_notebook.py speaker_verification.ipynb --print-install) \
    nbclient nbformat ipykernel
python ../../tools/run_notebook.py speaker_verification.ipynb
```

The notebooks pin a release rather than installing espnet from git. A course
notebook is worth having because it does the same thing in April that it does
today, and installing from `master` gives neither that nor a fast install: it
builds from source, and a break on `master` the night before class is a break
in class. Each release, the pin moves and the notebooks are run again.

Each notebook has its own workflow in `.github/workflows/`, all of them calling
`_run_notebook.yml`, and each runs every Sunday. That is why the badge in the
table above can be read per notebook: a red one names the notebook that broke,
not the batch it was in. What green means, and the three things it does not
cover, is in [`Demos/README.md`](../../Demos/README.md#what-green-means-and-what-it-does-not).

## What was changed

- The checkpoints, the points and the "submit a screenshot" instructions are
  gone, along with the empty answer cells they belonged to.
- Sections that record your voice in the browser are gone: they only work in
  Colab and they cannot be checked automatically.
- **Models come from the Hub where the Hub has them.** The enhancement
  notebook loaded three checkpoints from Google Drive; all three now load by
  tag from the [ESPnet organisation](https://huggingface.co/espnet), which
  also removes the dependency on git-lfs being installed. The two English to
  Spanish translation models are not published there, so those stay on Drive
  and the notebook says why.
- `espnet.asr.asr_utils.plot_spectrogram` and `torch_complex` were ESPnet1,
  which no longer exists. The spectrograms are drawn from `Stft` output.
- Paths are relative instead of rooted at `/content`, and `device="cuda"` is
  now conditional, so the notebooks run outside Colab and on CPU runtimes.
- `gdown --id X` became `gdown X`: the flag was removed in gdown 5.
- `pysndfile` is gone from the translation notebook. Nothing imported it and
  nothing declared it — it is a build from source that wants libsndfile's
  headers, and it was the only thing that stopped that notebook from
  installing on a clean runner.
- Outputs and widget state are cleared. One TensorBoard cell was carrying 9 MB
  of it.

## Credit

The notebooks were written by the course's instructors and teaching
assistants; each keeps its author line. They are kept here with the
demonstrations intact and the grading removed.
