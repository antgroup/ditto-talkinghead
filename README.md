<h2 align='center'>Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis</h2>

<div align='center'>
    <a href=""><strong>Tianqi Li</strong></a>
    ·
    <a href=""><strong>Ruobing Zheng</strong></a><sup>†</sup>
    ·
    <a href=""><strong>Minghui Yang</strong></a>
    ·
    <a href=""><strong>Jingdong Chen</strong></a>
    ·
    <a href=""><strong>Ming Yang</strong></a>
</div>
<div align='center'>
Ant Group
</div>
<br>
<div align='center'>
    <a href='https://arxiv.org/abs/2411.19509'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
    <a href='https://digital-avatar.github.io/ai/Ditto/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://huggingface.co/digital-avatar/ditto-talkinghead'><img src='https://img.shields.io/badge/Model-HuggingFace-yellow'></a>
    <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/badge/Code-GitHub-purple'></a>
    <!-- <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/github/stars/antgroup/ditto-talkinghead?style=social'></a> -->
    <a href='https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing'><img src='https://img.shields.io/badge/Demo-Colab-orange'></a>
</div>
<br>
<div align="center">
    <video style="width: 95%; object-fit: cover;" controls loop src="https://github.com/user-attachments/assets/ef1a0b08-bff3-4997-a6dd-62a7f51cdb40" muted="false"></video>
    <p>
    ✨  For more results, visit our <a href="https://digital-avatar.github.io/ai/Ditto/"><strong>Project Page</strong></a> ✨ 
    </p>
</div>


## 📌 Updates
* [2025.11.12] 🔥🔥 We noticed the community's enthusiasm for open-source training code. [Training code](https://github.com/antgroup/ditto-talkinghead/tree/train) is now available, since there have been multiple versions and limited time to organize, it may differ slightly from the paper version.
* [2025.07.11] 🔥 The [PyTorch model](#-pytorch-model) is now available.
* [2025.07.07] 🔥 Ditto is accepted by ACM MM 2025.
* [2025.01.21] 🔥 We update the [Colab](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing) demo, welcome to try it. 
* [2025.01.10] 🔥 We release our inference [codes](https://github.com/antgroup/ditto-talkinghead) and [models](https://huggingface.co/digital-avatar/ditto-talkinghead).
* [2024.11.29] 🔥 Our [paper](https://arxiv.org/abs/2411.19509) is in public on arxiv.

 
 ## 🔍 Overview
<!-- This is the **train branch**, containing code for **training the model**. For inference code, please switch to the [`main`](https://github.com/antgroup/ditto-talkinghead) branch. -->

This is the **inference branch**. For training code, please switch to the [`train`](https://github.com/antgroup/ditto-talkinghead/tree/train) branch.



## 🛠️ Installation

Tested Environment  
- System: Centos 7.2  
- GPU: A100  
- Python: 3.10  
- tensorRT: 8.6.1  


Clone the codes from [GitHub](https://github.com/antgroup/ditto-talkinghead):  
```bash
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead
```

### Conda
Create `conda` environment:
```bash
conda env create -f environment.yaml
conda activate ditto
```

### Pip
If you have problems creating a conda environment, you can also refer to our [Colab](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing). 
After correctly installing `pytorch`, `cuda` and `cudnn`, you only need to install a few packages using pip:
```bash
pip install \
    tensorrt==8.6.1 \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1
```

If you don't use `conda`, you may also need to install `ffmpeg` according to the [official website](https://www.ffmpeg.org/download.html).


## 📥 Download Checkpoints

Download checkpoints from [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead) and put them in `checkpoints` dir:
```bash
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

The `checkpoints` should be like:
```text
./checkpoints/
├── ditto_cfg
│   ├── v0.4_hubert_cfg_trt.pkl
│   └── v0.4_hubert_cfg_trt_online.pkl
├── ditto_onnx
│   ├── appearance_extractor.onnx
│   ├── blaze_face.onnx
│   ├── decoder.onnx
│   ├── face_mesh.onnx
│   ├── hubert.onnx
│   ├── insightface_det.onnx
│   ├── landmark106.onnx
│   ├── landmark203.onnx
│   ├── libgrid_sample_3d_plugin.so
│   ├── lmdm_v0.4_hubert.onnx
│   ├── motion_extractor.onnx
│   ├── stitch_network.onnx
│   └── warp_network.onnx
└── ditto_trt_Ampere_Plus
    ├── appearance_extractor_fp16.engine
    ├── blaze_face_fp16.engine
    ├── decoder_fp16.engine
    ├── face_mesh_fp16.engine
    ├── hubert_fp32.engine
    ├── insightface_det_fp16.engine
    ├── landmark106_fp16.engine
    ├── landmark203_fp16.engine
    ├── lmdm_v0.4_hubert_fp32.engine
    ├── motion_extractor_fp32.engine
    ├── stitch_network_fp16.engine
    └── warp_network_fp16.engine
```

- The `ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` is online config
- The `ditto_cfg/v0.4_hubert_cfg_trt.pkl` is offline config


## 🚀 Inference 

Run `inference.py`:

```shell
python inference.py \
    --data_root "<path-to-trt-model>" \
    --cfg_pkl "<path-to-cfg-pkl>" \
    --audio_path "<path-to-input-audio>" \
    --source_path "<path-to-input-image>" \
    --output_path "<path-to-output-mp4>" 
```

For example:

```shell
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4" 
```

❗Note:

We have provided the tensorRT model with `hardware-compatibility-level=Ampere_Plus` (`checkpoints/ditto_trt_Ampere_Plus/`). If your GPU does not support it, please execute the `cvt_onnx_to_trt.py` script to convert from the general onnx model (`checkpoints/ditto_onnx/`) to the tensorRT model.

```bash
python scripts/cvt_onnx_to_trt.py --onnx_dir "./checkpoints/ditto_onnx" --trt_dir "./checkpoints/ditto_trt_custom"
```

Then run `inference.py` with `--data_root=./checkpoints/ditto_trt_custom`.


## ⚡ PyTorch Model
*Based on community interest and to better support further development, we are now open-sourcing the PyTorch version of the model.*


We have added the PyTorch model and corresponding configuration files to the [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead). Please refer to [Download Checkpoints](#-download-checkpoints) to prepare the model files.

The `checkpoints` should be like:
```text
./checkpoints/
├── ditto_cfg
│   ├── ...
│   └── v0.4_hubert_cfg_pytorch.pkl
├── ...
└── ditto_pytorch
    ├── aux_models
    │   ├── 2d106det.onnx
    │   ├── det_10g.onnx
    │   ├── face_landmarker.task
    │   ├── hubert_streaming_fix_kv.onnx
    │   └── landmark203.onnx
    └── models
        ├── appearance_extractor.pth
        ├── decoder.pth
        ├── lmdm_v0.4_hubert.pth
        ├── motion_extractor.pth
        ├── stitch_network.pth
        └── warp_network.pth
```

To run inference, execute the following command:

```shell
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4" \
    --device auto
```

### Apple Silicon Mac

Apple Silicon Macs are supported through the PyTorch checkpoint path. CUDA and TensorRT are NVIDIA-only and are not available on macOS, so use `checkpoints/ditto_pytorch` with `--device auto` or `--device mps`.

Install dependencies with `uv` and Python 3.10. The Apple Silicon/PyTorch runtime dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

```bash
brew install uv ffmpeg git-lfs
git lfs install

uv python install 3.10
uv sync
```

Run inference:

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4" \
    --device auto
```

For a faster low-quality Apple Silicon preview, reduce diffusion steps and source size:

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result_fast_mac.mp4" \
    --device auto \
    --fast_mac
```

You can tune the tradeoff manually with `--sampling_timesteps`, `--max_size`, and `--online_mode`.

The final video with audio is written to `tmp/result.mp4`. The temporary file `tmp/result.mp4.tmp.mp4` contains only video frames and is silent until the final `ffmpeg` mux step completes. Apple Silicon inference is expected to be slower than the CUDA/TensorRT path.


## 📧 Acknowledgement
Our implementation is based on [S2G-MDDiffusion](https://github.com/thuhcsi/S2G-MDDiffusion) and [LivePortrait](https://github.com/KwaiVGI/LivePortrait). Thanks for their remarkable contribution and released code! If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.

## ⚖️ License
This repository is released under the Apache-2.0 license as found in the [LICENSE](LICENSE) file.

## 📚 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{li2024ditto,
    title={Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis},
    author={Li, Tianqi and Zheng, Ruobing and Yang, Minghui and Chen, Jingdong and Yang, Ming},
    journal={arXiv preprint arXiv:2411.19509},
    year={2024}
}
```


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/ditto-talkinghead&type=Date)](https://www.star-history.com/#antgroup/ditto-talkinghead&Date)
