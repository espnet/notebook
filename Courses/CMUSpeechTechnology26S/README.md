# CMUSpeechTechnology26S

CMU 11492/11692/18495, *Speech Technology for Conversational AI*, Spring 2026.

The demonstration notebooks from the course, kept here so that they keep
working: the graded exercises are removed, the installs point at released
ESPnet rather than at anyone's fork, and each one is run before it is changed.

They are meant to be opened in Colab, and they run on CPU.

The course's two fine-tuning notebooks are not here yet: they are written
against an ESPnet3 data API that has since changed, and both stop at
`recipe_dir must be set when data_src is None`. They come back when they are
updated to the current one.

| Notebook | What it does |
|---|---|
| [`speaker_verification.ipynb`](speaker_verification.ipynb) | Speaker embeddings with ESPnet-SPK, verification by cosine similarity, and a simple diarization of a two-speaker mixture |
| [`speech_enhancement.ipynb`](speech_enhancement.ipynb) | Enhancement of real noisy speech, separation of a two-speaker mixture, and scoring with VERSA and a pretrained ASR model |
| [`neural_codec.ipynb`](neural_codec.ipynb) | Three pretrained neural codecs, the bitrate trade when streams are dropped, and VERSA scores for each |
| [`speech_translation.ipynb`](speech_translation.ipynb) | Offline and simultaneous speech translation with ESPnet-ST-v2, scored with BLEU and SimulEval's latency metrics |
| [`text_to_speech.ipynb`](text_to_speech.ipynb) | Single-speaker and multi-speaker synthesis, text2wav against text2mel plus a vocoder, and VERSA scores |

## Running them outside Colab

`run_notebook.py` executes one of them here: it skips the install cells, since
the packages are already present, and replaces `wget` and `tar` with their
Python equivalents.

```sh
pip install "espnet==202610.post1" nbclient nbformat ipykernel librosa \
    scikit-learn matplotlib
python run_notebook.py speaker_verification.ipynb
```

The notebooks pin that release rather than installing espnet from git. A course
notebook is worth having because it does the same thing in April that it does
today, and installing from `master` gives neither that nor a fast install: it
builds from source, and a break on `master` the night before class is a break
in class. Each release, the pin moves and the notebooks are run again.

`.github/workflows/run_notebooks.yml` runs `speaker_verification.ipynb` every
Sunday, so a notebook that stops working is noticed here rather than in class.

**One of the five, not all five.** It is the one that fits a free runner: no
GPU, nothing that has to be built, and it finishes inside the timeout. The
others are each blocked on something of their own — `text_to_speech` builds a
vocoder from source, `neural_codec` and `speech_translation` pin an old numpy
and TensorFlow against the rest of the environment, `speech_enhancement` runs
VERSA over several models. Until one of those is sorted out, the other four are
checked by running them by hand, which is a worse guarantee and should be said
plainly rather than implied.

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
- Outputs and widget state are cleared. One TensorBoard cell was carrying 9 MB
  of it.

## Credit

The notebooks were written by the course's instructors and teaching
assistants; each keeps its author line. They are kept here with the
demonstrations intact and the grading removed.
