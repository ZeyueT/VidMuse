# VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling 🎶

[![arXiv](https://img.shields.io/badge/arXiv-2406.04321-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2406.04321)  [![githubio](https://img.shields.io/badge/GitHub.io-Project page-blue?logo=Github&style=flat-square)](https://vidmuse.github.io/)

**This is the official repository for "[VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling](https://arxiv.org/pdf/2406.04321)".** 



## 📺 Demo Video
[![Watch the video](https://github.com/ZeyueT/VidMuse/assets/126848881/ec71637c-9d3b-4cc5-8e14-0b76929f647a)](https://www.youtube.com/watch?v=hXjEFv5s9Zk)


## 🔆 Abstract

In this work, we systematically study music generation conditioned solely on the video.
First, we present the large-scale dataset V2M, which comprises 190K video-music pairs and includes various genres such as movie trailers, advertisements, and documentaries.
Furthermore, we propose VidMuse, a simple framework for generating music aligned with video inputs. VidMuse stands out by producing high-fidelity music that is both acoustically and semantically aligned with the video. By incorporating local and global visual cues, VidMuse enables the creation of musically coherent audio tracks that consistently match the video content through Long-Short-Term modeling. Through extensive experiments, VidMuse outperforms existing models in terms of audio quality, diversity, and audio-visual alignment. 

## 🔆 Data Construction
![data_pipeline](https://github.com/ZeyueT/VidMuse/assets/126848881/91562024-3feb-4d56-8f1f-5c58a79187ab)
**Dataset Construction.** To ensure data quality, V2M goes through rule-based coarse filtering and content-based fine-grained filtering. Music source separation is applied to remove speech and singing signals in the audio. After processing, human experts curate the benchmark subset, while the remaining data is used as the pretraining dataset. The pretrain data is then refined using Audio-Visual Alignment Ranking to select the finetuning dataset.

## 🔆 Method
![method](https://github.com/ZeyueT/VidMuse/assets/126848881/25c54387-2136-4d61-956c-abf07146bea6)
**Overview of the VidMuse Framework.**

## ⚙️ Code & Datasets
Our code and datasets will come soon.

## 🤗 Citation

```
@article{tian2024vidmuse,
  title={VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling},
  author={Tian, Zeyue and Liu, Zhaoyang and Yuan, Ruibin and Pan, Jiahao and Huang, Xiaoqiang and Liu, Qifeng and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2406.04321},
  year={2024}
}
```

<hr>
