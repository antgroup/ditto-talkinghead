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
    <a href='https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing'><img src='https://img.shields.io/badge/Demo-Colab-orange'></a>
    <!-- <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/github/stars/antgroup/ditto-talkinghead?style=social'></a> -->
</div>
<br>
<div align="center">
    <video style="width: 95%; object-fit: cover;" controls loop src="https://github.com/user-attachments/assets/ef1a0b08-bff3-4997-a6dd-62a7f51cdb40" muted="false"></video>
    <p>
    ✨  For more results, visit our <a href="https://digital-avatar.github.io/ai/Ditto/"><strong>Project Page</strong></a> ✨ 
    </p>
</div>


## 📌 Updates
* [2025.01.21] 🔥 We update the [Colab](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing) demo, welcome to try it. 
* [2025.01.10] 🔥 We release our inference [codes](https://github.com/antgroup/ditto-talkinghead) and [models](https://huggingface.co/digital-avatar/ditto-talkinghead).
* [2024.11.29] 🔥 Our [paper](https://arxiv.org/abs/2411.19509) is in public on arxiv.

 

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
python script/cvt_onnx_to_trt.py --onnx_dir "./checkpoints/ditto_onnx" --trt_dir "./checkpoints/ditto_trt_custom"
```

Then run `inference.py` with `--data_root=./checkpoints/ditto_trt_custom`.




## Docker + nvidia runtime container

Build the container:
```shell
docker build -t ditto-talkinghead .
```
Run the container with GPU support:

```shell
docker run --gpus all -v $(pwd)/output:/app/output ditto-talkinghead
```
Or to run with custom input files:

```shell
docker run --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  ditto-talkinghead \
  python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "/app/input/your_audio.wav" \
    --source_path "/app/input/your_image.png" \
    --output_path "/app/output/result.mp4"
```

To run the container you need nvidia runtime container

# Setup the package repository and GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package listing and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure the Docker daemon to recognize NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker

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
