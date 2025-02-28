# VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling 🎶

**This project has been accepted to CVPR 2025! 🚀🚀🚀**

[![arXiv](https://img.shields.io/badge/arXiv-2406.04321-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2406.04321)   [![githubio](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://vidmuse.github.io/)

**This is the official repository for "[VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling](https://arxiv.org/pdf/2406.04321)".**

## 📺 Demo Video

[![Watch the video](https://github.com/user-attachments/assets/54107f99-2399-49ea-aa8e-131b7170d617)](https://www.youtube.com/watch?v=DbZbzcVI6qg)



## ✨ Abstract

In this work, we systematically study music generation conditioned solely on the video.
First, we present the large-scale dataset V2M, which comprises 190K video-music pairs and includes various genres such as movie trailers, advertisements, and documentaries.
Furthermore, we propose VidMuse, a simple framework for generating music aligned with video inputs. VidMuse stands out by producing high-fidelity music that is both acoustically and semantically aligned with the video. By incorporating local and global visual cues, VidMuse enables the creation of musically coherent audio tracks that consistently match the video content through Long-Short-Term modeling. Through extensive experiments, VidMuse outperforms existing models in terms of audio quality, diversity, and audio-visual alignment.

## ✨ Data Construction

![data_pipeline](https://github.com/ZeyueT/VidMuse/assets/126848881/91562024-3feb-4d56-8f1f-5c58a79187ab)
**Dataset Construction.** To ensure data quality, V2M goes through rule-based coarse filtering and content-based fine-grained filtering. Music source separation is applied to remove speech and singing signals in the audio. After processing, human experts curate the benchmark subset, while the remaining data is used as the pretraining dataset. The pretrain data is then refined using Audio-Visual Alignment Ranking to select the finetuning dataset.

## ✨ Method

<p align="center">
  <img src="https://github.com/ZeyueT/VidMuse/assets/126848881/25c54387-2136-4d61-956c-abf07146bea6" alt="method">
</p>

<p align="center"><strong>Overview of the VidMuse Framework.</strong></p>

## 🛠️ Environment Setup

- Create Anaconda Environment:
  AudioCraft requires Python 3.9, PyTorch 2.1.0.
  
  ```bash
  git clone https://github.com/ZeyueT/VidMuse.git; cd VidMuse
  ```
  
  ```bash
  conda create -n VidMuse python=3.9
  conda activate VidMuse
  pip install git+https://github.com/ZeyueT/VidMuse.git
  ```
- Install ffmpeg:
  
  ```bash
  sudo apt-get install ffmpeg
  # Or if you are using Anaconda or Miniconda
  conda install "ffmpeg<5" -c conda-forge
  ```

## 🔮 Pretrained Weights

- Please download the pretrained Audio Compression checkpoint [compression_state_dict.bin](https://drive.google.com/file/d/1o1JznkelMiGizV5RhPIamsi5d5VBs4G_/view?usp=sharing) and VidMuse model checkpoint [state_dict.bin](https://drive.google.com/file/d/1RfVrg3d_UFKL2we8JaVqcnhK4P6TIe_9/view?usp=sharing), put them into the directory `'./model'`. (The VidMuse model is trained with our private dataset.)
  ```bash
  mkdir model
  mv compression_state_dict.bin ./model
  mv state_dict.bin ./model
  ```

## 🎞 Web APP

- Use the gradio demo locally by running:
  
  ```bash
  python -m demos.VidMuse_app --share
  ```
  
  In the Gradio application, the Model Path field is used to specify the location of the model files. To correctly load the model, set the Model Path to `'./model'`.

## 🔥 Training

- Build data.jsonl file:
  
  ```bash
  python egs/V2M/build_data_jsonl.py
  ```
- Start training:
  
  ```bash
  bash train.sh
  ```

## 📥 Importing / Exporting models

- To export the trained model, use the following script:
  
  ```python
  import os
  import torch
  from audiocraft.utils import export
  from audiocraft import train

  # Define codec_model
  codec_model = 'facebook/encodec_32khz'
  xp = train.main.get_xp_from_sig('SIG')

  model_save_path = './model'

  # Export model
  export.export_lm(xp.folder / 'checkpoint.th', model_save_path + '/state_dict.bin')
  export.export_pretrained_compression_model(codec_model, model_save_path + '/compression_state_dict.bin')
  ```

## 🎯 Infer

- Before running the inference script, make sure to define the following parameters in `infer.sh`:
  - `model_path`: Path to the model directory. This is where the model files are stored. Default is `'./model'`.
  - `video_dir`: Directory containing the videos for inference. This is where the input videos are located. Default is `'./dataset/example/infer'`.
  - `output_dir`: Directory where the output generated music will be saved. Default is `'./result/'`.

- Run the inference using the following script:

  ```bash
  bash infer.sh
  ```

## 🧱 Dataset & Dataset Construction

- To be released...

## 🤗 Acknowledgement

- [Audiocraft](https://github.com/facebookresearch/audiocraft): the codebase we built upon.

## 🚀 Citation

If you find our work useful, please consider citing:

```
@article{tian2024vidmuse,
  title={Vidmuse: A simple video-to-music generation framework with long-short-term modeling},
  author={Tian, Zeyue and Liu, Zhaoyang and Yuan, Ruibin and Pan, Jiahao and Liu, Qifeng and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2406.04321},
  year={2024}
}
```

## 📭 Contact

If you have any comments or questions, feel free to contact Zeyue Tian(ztianad@connect.ust.hk), Zhaoyang Liu(zyliumy@gmail.com).

## License

Please follow [CC-BY-NC](./LICENSE).

<hr>

