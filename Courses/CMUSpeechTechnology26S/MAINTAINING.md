# Maintaining these notebooks

What was changed from the course originals, and what a reader would otherwise
have to work out by running them. The notebooks themselves are the
documentation for what they teach; this file is for whoever keeps them
running.

## What the badges check

Each notebook has a workflow of its own and runs every Sunday, cell by cell,
against the release it pins, so a badge is that notebook and nothing else: a
red one names the notebook that broke rather than the batch it was in.

`speech_recognition` is the exception. Fine-tuning does not fit a free
runner, so its workflow shrinks the run through the environment - eight
utterances instead of a split, two steps instead of three hundred - and its
workflow is named `speech_recognition (short run)`, which is what the badge
then says. **Green there means the notebook installs, finds its data and
still agrees with ESPnet3's API. It does not mean the fine-tuning worked, or
that the numbers printed in the notebook come out again.** Nothing in CI
checks those. A reader who opens it gets the full run, and the notebook says
the same in its first cell.

What the environment variables are, and which each workflow sets, is in
`.github/workflows/`.

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
- **wsj0-2mix → Libri2Mix.** The separation section used a mixture from
  wsj0-2mix, which is built on WSJ0 and cannot be published, fetched from
  Drive. It now separates a **Libri2Mix** mixture instead, and the model
  changed with the data:
  [`espnet/anogkongda_librimix_enh_train_raw_valid.si_snr.ave`](https://huggingface.co/espnet/anogkongda_librimix_enh_train_raw_valid.si_snr.ave),
  a Conv-TasNet trained on Libri2Mix. Libri2Mix is built on LibriSpeech, which
  is CC BY 4.0, so the two sources could be published beside the model; the
  notebook builds the mixture from them the way LibriMix does — the gains the
  official metadata gives for that pair, resampled to 8 kHz, truncated to the
  shorter source. It is the real test mixture
  `7729-102255-0031_2094-142345-0028`, not an improvisation.

  Having the sources means the separation is **scored** rather than only
  listened to: SI-SNR 3.24 → 16.21 dB for one speaker and −3.28 → 12.88 dB for
  the other. The ASR that follows moved to a LibriSpeech model for the same
  reason, and it reads the separated streams almost perfectly while the
  mixture comes out as nonsense, which is the point of the section.

  The noisy half of Libri2Mix adds WHAM! noise, CC BY-NC 4.0, and is not used.

  That model did not load at all before this: a checkpoint from before May
  2023 cannot be built by current espnet, because `TCNSeparator` defaults to a
  layout its weights predate. Its config now records the layout it was trained
  with, and so do four others in the organisation that were unloadable for the
  same reason.

## What was changed from the course originals

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
- **`speech_recognition` follows ESPnet3's current data API.** It defined a
  `torch.utils.data.Dataset` in the notebook and named it in the config as
  `_target_: __main__.FLEURSDataset`. That is gone:
  `DataOrganizer` resolves every entry through `load_dataset_module()`, which
  imports `<recipe_dir>/dataset/__init__.py` and expects a class called
  `Dataset`, with the config passing its arguments as `data_src_args`. The
  notebook now writes that module with `%%writefile` and points `recipe_dir`
  at it, which is the layout a recipe in `egs3/` uses.
- It no longer names a checkpoint by step number. `step300.ckpt` is only right
  while nothing above it changes; it now loads the last checkpoint the trainer
  wrote.
- **The weekly run of it is a smaller run.** A free runner cannot fine-tune at
  full size, so its workflow sets a few environment variables - eight
  utterances, two steps instead of three hundred - and says which. The badge
  means the notebook installs, finds its data and still agrees with ESPnet3's
  API; it does not mean the numbers printed in it were reproduced. Open it and
  you get the full run.
- The course's second fine-tuning notebook, which fine-tuned on spoken digits
  and then compared CTC decoding against the beam search, is not here. One
  fine-tuning notebook is enough to maintain, and this is the one.
