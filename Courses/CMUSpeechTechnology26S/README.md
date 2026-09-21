# CMU Speech Technology 26S

CMU 11492/11692/18495, *Speech Technology for Conversational AI*, Spring 2026.

The demonstration notebooks from the course. Open one in Colab and run it top
to bottom; they run on CPU.

| Notebook | | What it does |
|---|---|---|
| [`speaker_verification.ipynb`](speaker_verification.ipynb) | [![speaker_verification](https://github.com/espnet/notebook/actions/workflows/speaker_verification.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speaker_verification.yml) | Speaker embeddings with ESPnet-SPK, verification, and a simple diarization |
| [`speech_enhancement.ipynb`](speech_enhancement.ipynb) | [![speech_enhancement](https://github.com/espnet/notebook/actions/workflows/speech_enhancement.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_enhancement.yml) | Enhancement and separation, scored with VERSA and a pretrained ASR model |
| [`text_to_speech.ipynb`](text_to_speech.ipynb) | [![text_to_speech](https://github.com/espnet/notebook/actions/workflows/text_to_speech.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/text_to_speech.yml) | Single-speaker and multi-speaker synthesis, and VERSA scores |
| [`neural_codec.ipynb`](neural_codec.ipynb) | [![neural_codec](https://github.com/espnet/notebook/actions/workflows/neural_codec.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/neural_codec.yml) | Three pretrained neural codecs and the bitrate trade between them |
| [`speech_translation.ipynb`](speech_translation.ipynb) | [![speech_translation](https://github.com/espnet/notebook/actions/workflows/speech_translation.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_translation.yml) | Offline and simultaneous speech translation with ESPnet-ST-v2 |
| [`speech_recognition.ipynb`](speech_recognition.ipynb) | [![speech_recognition](https://github.com/espnet/notebook/actions/workflows/speech_recognition.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/speech_recognition.yml) | Fine-tune OWSM on one language of FLEURS with the ESPnet3 trainer |

## Credit

The notebooks were written by the course's instructors and teaching
assistants; each keeps its author line. They are kept here with the
demonstrations intact and the grading removed.

## For maintainers

[`MAINTAINING.md`](MAINTAINING.md) has what the badges check and what they do
not, how to run these outside Colab, what was changed from the course
originals, and which recordings had to be replaced and why.
