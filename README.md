# ESPnet Notebooks

## Demos

### ASR (Speech recognition)

- [`asr_realtime_demo.ipynb`](Demos/ASR/asr_realtime_demo.ipynb): ASR realtime inference with various pre-trained models.
- [`asr_transfer_learning_demo.ipynb`](Demos/ASR/asr_transfer_learning_demo.ipynb): Demo on how to use pre-trained ASR models for fine-tuning.
- [`streaming_asr_demo.ipynb`](Demos/ASR/streaming_asr_demo.ipynb): Streaming ASR realtime inference with pre-trained models.

### SE (Speech enhancement/separation)

- [`se_demo.ipynb`](Demos/SE/se_demo.ipynb): Speech enhancement/separation inference with various pre-trained models.
- [`se_demo_for_waspaa_2021.ipynb`](Demos/SE/se_demo_for_waspaa_2021.ipynb): WASPAA2021 version of ESPnet-SE demo.

### SLU (Spoken language understanding)

- [`2pass_slu_demo.ipynb`](Demos/SLU/2pass_slu_demo.ipynb): Two pass spoken language understanding pre-trained model examples.

### TTS (Text-to-speech)

- [`tts_realtime_demo.ipynb`](Demos/TTS/tts_realtime_demo.ipynb): TTS realtime inference with various pre-trained models.

### Other utilities

- [`onnx_conversion_demo.ipynb`](Demos/Others/onnx_conversion_demo.ipynb): How to convert ESPnet models into ONNX format.


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
