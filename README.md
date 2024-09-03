# ESPnet Notebooks

## Demo

### ASR (Speech recognition)

- [`asr_realtime_demo.ipynb`](ESPnet2/Demo/ASR/asr_realtime_demo.ipynb): ASR realtime inference with various pre-trained models.
- [`asr_transfer_learning_demo.ipynb`](ESPnet2/Demo/ASR/asr_transfer_learning_demo.ipynb): Demo on how to use pre-trained ASR models for fine-tuning.
- [`streaming_asr_demo.ipynb`](ESPnet2/Demo/ASR/streaming_asr_demo.ipynb): Streaming ASR realtime inference with pre-trained models.

### SE (Speech enhancement/separation)

- [`se_demo.ipynb`](ESPnet2/Demo/SE/se_demo.ipynb): Speech enhancement/separation inference with various pre-trained models.
- [`se_demo_for_waspaa_2021.ipynb`](ESPnet2/Demo/SE/se_demo_for_waspaa_2021.ipynb): WASPAA2021 version of ESPnet-SE demo.

### SLU (Spoken language understanding)

- [`2pass_slu_demo.ipynb`](ESPnet2/Demo/SLU/2pass_slu_demo.ipynb): Two pass spoken language understanding pre-trained model examples.

### TTS (Text-to-speech)

- [`tts_realtime_demo.ipynb`](ESPnet2/Demo/TTS/tts_realtime_demo.ipynb): TTS realtime inference with various pre-trained models.

### Other utilities

- [`onnx_conversion_demo.ipynb`](ESPnet2/Demo/Others/onnx_conversion_demo.ipynb): How to convert ESPnet models into ONNX format.


## ESPnet-EZ

### ASR (Speech recognition)
- [`train_from_scratch.ipynb`](ESPnetEZ/ASR/train_from_scratch.ipynb): Training an ASR model with ESPnet-EZ on LibriSpeech-100.
- [`ASR_finetune_owsm.ipynb`](ESPnetEZ/ASR/ASR_finetune_owsm.ipynb): Fine-tuning the weakly-supervised model (OWSM) with ESPnet-EZ on custom dataset.

### ST (Speech-to-text translation)
- [`integrate_huggingface.ipynb`](ESPnetEZ/ST/integrate_huggingface.ipynb): Integrating the weakly-supervised model (OWSM) and huggingface's pre-trained language model with ESPnet-EZ on MuST-C-v2.
- [`ST_finetune_owsm.ipynb`](ESPnetEZ/ST/ST_finetune_owsm.ipynb): Fine-tuning the weakly-supervised model (OWSM) with ESPnet-EZ on MuST-C-v2.

### SLU (Spoken language understanding)
- [`SLU_finetune_owsm.ipynb`](ESPnetEZ/SLU/SLU_finetune_owsm.ipynb): Fine-tuning the weakly-supervised model (OWSM) with ESPnet-EZ on SLURP.


## Course

### CMU SpeechProcessing Spring2023

- [`assignment0_data-prep.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment0_data-prep.ipynb): Course assignment on how to prepare ESPnet-format data.
- [`assignment1_espnet-tutorial.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment1_espnet-tutorial.ipynb): A simplified version of previous year's new task tutorial.
- [`assignemnt3_spk.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment3_spk.ipynb): Examples of using ESPnet to extract speaker embeddings and conduct speaker recognition.
- [`assignment4_ssl.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment4_ssl.ipynb): Exploration on using self-supervised speech representation to ESPnet ASR training.
- [`assignment5_st.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment5_st.ipynb): Examples of state-of-the-art speech translation models in ESPnet.
- [`assignment6_slu.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment6_slu.ipynb): Examples of state-of-the-art spoken language understanding models in ESPnet.
- [`assignment7_se.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment7_se.ipynb): Examples of state-of-the-art speech enhancement/separation in ESPnet.
- [`assignment8_tts.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment8_tts.ipynb): A student version of espnet2-tts realtime demonstration.
- [`s2st_demo.ipynb`](ESPnet2/Course/CMU_SpeechProcessing_Spring2023/s2st_demo.ipynb): An example of existing speech-to-speech translation model for ESPnet.

### CMU SpeechRecognition Fall2022

- [`recipe_tutorial.ipynb`](ESPnet2/Course/CMU_SpeechRecognition_Fall2022/recipe_tutorial.ipynb): A general tutorial of stage-by-stage explanation of ESPnet2 recipes (with new functions).
- [`new_task_tutorial.ipynb`](ESPnet2/Course/CMU_SpeechRecognition_Fall2022/new_task_tutorial.ipynb): A tutorial on how to add new models/tasks to ESPnet framework.

### CMU SpeechRecognition Fall2021

- [`general_tutorial.ipynb`](ESPnet2/Course/CMU_SpeechRecognition_Fall2021/general_tutorial.ipynb): A general tutorial of stage-by-stage explanation of ESPnet2 recipes.

## ESPnet1 (Legacy)

- [`asr_library.ipynb`](ESPnet1/asr_library.ipynb): Speech recognition library explanation with network training.
- [`asr_recipe.ipynb`](ESPnet1/asr_recipe.ipynb): Speech recognition recipe explanation.
- [`pretrained.ipynb`](ESPnet1/pretrained.ipynb): Tutorial on how to use pre-trained models.
- [`st_demo.ipynb`](ESPnet1/st_demo.ipynb): Speech translation demonstration with a TTS model to achieve speech-to-speech translation.
- [`tts_realtime_demo.ipynb`](ESPnet1/tts_realtime_demo.ipynb): TTS demonstration with different pre-trained TTS models.
- [`tts_recipe.ipynb`](ESPnet1/tts_recipe.ipynb): Stage explanation for TTS recipes.
