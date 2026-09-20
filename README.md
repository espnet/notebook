# ESPnet Notebooks

[![Checked weekly](https://github.com/espnet/notebook/actions/workflows/run_notebooks.yml/badge.svg)](https://github.com/espnet/notebook/actions/workflows/run_notebooks.yml)

Notebooks for [ESPnet](https://github.com/espnet/espnet): short demos of what
the toolkit does, and the material from the CMU speech courses.

The badge is the demos below, executed cell by cell every Sunday against the
release they pin. Nothing goes in that list until it runs there —
[what green means, and what it does not](Demos/README.md#what-green-means-and-what-it-does-not).

## Demos

One per task, flat in [`Demos/`](Demos), each short enough to read in a sitting.

| Notebook | What it does |
|---|---|
| [`asr_demo.ipynb`](Demos/asr_demo.ipynb) | Transcribe speech with OWSM-CTC, and let it work out the language |
| [`asr_streaming_demo.ipynb`](Demos/asr_streaming_demo.ipynb) | Watch the words appear while the audio is still arriving |
| [`st_demo.ipynb`](Demos/st_demo.ipynb) | Translate English speech into German, French and Chinese — the same model |
| [`tts_demo.ipynb`](Demos/tts_demo.ipynb) | Type a sentence, hear it spoken — one English voice, then 128 of them |
| [`enh_demo.ipynb`](Demos/enh_demo.ipynb) | Pull speech out of noise, and measure how much it helped |
| [`spk_demo.ipynb`](Demos/spk_demo.ipynb) | Turn a voice into a vector, and score two recordings against each other |
| [`codec_demo.ipynb`](Demos/codec_demo.ipynb) | Compress a waveform to a few integers a frame, rebuild it, count the bits |
| [`sds_demo.ipynb`](Demos/sds_demo.ipynb) | Speech in, speech out, with a language model thinking in between |

[`Demos/unmaintained/`](Demos/unmaintained) holds what was here before — the
oldest from 2021, most of it no longer running, kept because some of it is
still the only written record of how a thing was done. **It carries no badge
and nothing checks it.**

[`Demos/README.md`](Demos/README.md) has the naming rule and what a demo owes
the reader.

## Courses

### CMUSpeechTechnology26S

CMU 11492/11692/18495, *Speech Technology for Conversational AI*, Spring 2026.

Maintained: the graded exercises are removed and each notebook is run before
it is changed. See [`Courses/CMUSpeechTechnology26S/`](Courses/CMUSpeechTechnology26S) for the
full list and what was changed.

- [`speaker_verification.ipynb`](Courses/CMUSpeechTechnology26S/speaker_verification.ipynb): Speaker embeddings with ESPnet-SPK, verification, and a simple diarization.
- [`speech_enhancement.ipynb`](Courses/CMUSpeechTechnology26S/speech_enhancement.ipynb): Enhancement and separation, scored with VERSA and a pretrained ASR model.
- [`neural_codec.ipynb`](Courses/CMUSpeechTechnology26S/neural_codec.ipynb): Three pretrained neural codecs and the bitrate trade between them.
- [`speech_translation.ipynb`](Courses/CMUSpeechTechnology26S/speech_translation.ipynb): Offline and simultaneous speech translation with ESPnet-ST-v2.
- [`text_to_speech.ipynb`](Courses/CMUSpeechTechnology26S/text_to_speech.ipynb): Single-speaker and multi-speaker synthesis, and VERSA scores.

### CMUSpeechProcessing23S

CMU 11492/11692, Spring 2023.

- [`assignment1_espnet-tutorial.ipynb`](Courses/CMUSpeechProcessing23S/assignment1_espnet-tutorial.ipynb): A simplified version of previous year's new task tutorial.
- [`assignemnt3_spk.ipynb`](Courses/CMUSpeechProcessing23S/assignment3_spk.ipynb): Examples of using ESPnet to extract speaker embeddings and conduct speaker recognition.
- [`assignment4_ssl.ipynb`](Courses/CMUSpeechProcessing23S/assignment4_ssl.ipynb): Exploration on using self-supervised speech representation to ESPnet ASR training.
- [`assignment5_st.ipynb`](Courses/CMUSpeechProcessing23S/assignment5_st.ipynb): Examples of state-of-the-art speech translation models in ESPnet.
- [`assignment6_slu.ipynb`](Courses/CMUSpeechProcessing23S/assignment6_slu.ipynb): Examples of state-of-the-art spoken language understanding models in ESPnet.
- [`assignment7_se.ipynb`](Courses/CMUSpeechProcessing23S/assignment7_se.ipynb): Examples of state-of-the-art speech enhancement/separation in ESPnet.
- [`assignment8_tts.ipynb`](Courses/CMUSpeechProcessing23S/assignment8_tts.ipynb): A student version of espnet2-tts realtime demonstration.
- [`s2st_demo.ipynb`](Courses/CMUSpeechProcessing23S/s2st_demo.ipynb): An example of existing speech-to-speech translation model for ESPnet.

### CMUSpeechRecognition22F

CMU 11751/18781, Fall 2022.

- [`recipe_tutorial.ipynb`](Courses/CMUSpeechRecognition22F/recipe_tutorial.ipynb): A general tutorial of stage-by-stage explanation of ESPnet2 recipes (with new functions).
- [`new_task_tutorial.ipynb`](Courses/CMUSpeechRecognition22F/new_task_tutorial.ipynb): A tutorial on how to add new models/tasks to ESPnet framework.

### CMUSpeechRecognition21F

CMU 11751/18781, Fall 2021.

- [`general_tutorial.ipynb`](Courses/CMUSpeechRecognition21F/general_tutorial.ipynb): A general tutorial of stage-by-stage explanation of ESPnet2 recipes.
