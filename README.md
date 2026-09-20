# ESPnet Notebooks


Notebooks for [ESPnet](https://github.com/espnet/espnet): short demos of what
the toolkit does, and the material from the CMU speech courses.

Each badge is that notebook, executed cell by cell every Sunday against the
release it pins. Nothing joins a table until it runs there —
[what green means, and what it does not](Demos/README.md#what-green-means-and-what-it-does-not).

## Demos

One per task, flat in [`Demos/`](Demos), each short enough to read in a sitting.

| Notebook | | What it does |
|---|---|---|
| [`asr_demo.ipynb`](Demos/asr_demo.ipynb) | [![asr_demo](https://github.com/espnet/notebook/actions/workflows/asr_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/asr_demo.yml) | Transcribe speech with OWSM-CTC, and let it work out the language |
| [`asr_streaming_demo.ipynb`](Demos/asr_streaming_demo.ipynb) | [![asr_streaming_demo](https://github.com/espnet/notebook/actions/workflows/asr_streaming_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/asr_streaming_demo.yml) | Watch the words appear while the audio is still arriving |
| [`st_demo.ipynb`](Demos/st_demo.ipynb) | [![st_demo](https://github.com/espnet/notebook/actions/workflows/st_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/st_demo.yml) | Translate English speech into German, French and Chinese — the same model |
| [`tts_demo.ipynb`](Demos/tts_demo.ipynb) | [![tts_demo](https://github.com/espnet/notebook/actions/workflows/tts_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/tts_demo.yml) | Type a sentence, hear it spoken — one English voice, then 128 of them |
| [`enh_demo.ipynb`](Demos/enh_demo.ipynb) | [![enh_demo](https://github.com/espnet/notebook/actions/workflows/enh_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/enh_demo.yml) | Pull speech out of noise, and measure how much it helped |
| [`spk_demo.ipynb`](Demos/spk_demo.ipynb) | [![spk_demo](https://github.com/espnet/notebook/actions/workflows/spk_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/spk_demo.yml) | Turn a voice into a vector, and score two recordings against each other |
| [`codec_demo.ipynb`](Demos/codec_demo.ipynb) | [![codec_demo](https://github.com/espnet/notebook/actions/workflows/codec_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/codec_demo.yml) | Compress a waveform to a few integers a frame, rebuild it, count the bits |
| [`sds_demo.ipynb`](Demos/sds_demo.ipynb) | [![sds_demo](https://github.com/espnet/notebook/actions/workflows/sds_demo.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/sds_demo.yml) | Speech in, speech out, with a language model thinking in between |

[`Demos/unmaintained/`](Demos/unmaintained) holds what was here before — the
oldest from 2021, most of it no longer running, kept because some of it is
still the only written record of how a thing was done. **It carries no badge
and nothing checks it.**

[`Demos/README.md`](Demos/README.md) has the naming rule and what a demo owes
the reader.

## Courses

### CMU Speech Technology 26S

CMU 11492/11692/18495, *Speech Technology for Conversational AI*, Spring 2026 —
the demonstration notebooks, with the graded exercises removed. In
[`Courses/CMUSpeechTechnology26S/`](Courses/CMUSpeechTechnology26S), which also
says how each is kept running — including why the two fine-tuning badges are
labelled *(short run)* and what they do not check.

| Notebook | | What it does |
|---|---|---|
| [`speaker_verification.ipynb`](Courses/CMUSpeechTechnology26S/speaker_verification.ipynb) | [![speaker_verification](https://github.com/espnet/notebook/actions/workflows/speaker_verification.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speaker_verification.yml) | Speaker embeddings with ESPnet-SPK, verification, and a simple diarization |
| [`speech_enhancement.ipynb`](Courses/CMUSpeechTechnology26S/speech_enhancement.ipynb) | [![speech_enhancement](https://github.com/espnet/notebook/actions/workflows/speech_enhancement.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_enhancement.yml) | Enhancement and separation, scored with VERSA and a pretrained ASR model |
| [`text_to_speech.ipynb`](Courses/CMUSpeechTechnology26S/text_to_speech.ipynb) | [![text_to_speech](https://github.com/espnet/notebook/actions/workflows/text_to_speech.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/text_to_speech.yml) | Single-speaker and multi-speaker synthesis, and VERSA scores |
| [`neural_codec.ipynb`](Courses/CMUSpeechTechnology26S/neural_codec.ipynb) | [![neural_codec](https://github.com/espnet/notebook/actions/workflows/neural_codec.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/neural_codec.yml) | Three pretrained neural codecs and the bitrate trade between them |
| [`speech_translation.ipynb`](Courses/CMUSpeechTechnology26S/speech_translation.ipynb) | [![speech_translation](https://github.com/espnet/notebook/actions/workflows/speech_translation.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_translation.yml) | Offline and simultaneous speech translation with ESPnet-ST-v2 |
| [`owsm_finetuning.ipynb`](Courses/CMUSpeechTechnology26S/owsm_finetuning.ipynb) | [![owsm_finetuning](https://github.com/espnet/notebook/actions/workflows/owsm_finetuning.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/owsm_finetuning.yml) | Fine-tune OWSM on one language of FLEURS with the ESPnet3 trainer — **badge is a short run**, see below |
| [`owsm_finetuning_ctc.ipynb`](Courses/CMUSpeechTechnology26S/owsm_finetuning_ctc.ipynb) | [![owsm_finetuning_ctc](https://github.com/espnet/notebook/actions/workflows/owsm_finetuning_ctc.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/owsm_finetuning_ctc.yml) | Fine-tune on a small spoken-digit corpus, then CTC against beam search — **badge is a short run**, see below |

### Earlier courses

[`Courses/`](Courses) also holds the material from CMU 11492/11692 Spring 2023,
and 11751/18781 Fall 2021 and Fall 2022. Nothing runs them and nothing has
checked them for years; they are kept as a record of how these things were
taught and done. Open one expecting to fix it before it works.
