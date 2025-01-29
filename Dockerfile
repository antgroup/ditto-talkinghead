# Start from NVIDIA TensorRT base image
FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch \
    tensorrt==8.6.1 \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv-python-headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1

# Clone the repository
RUN git clone https://github.com/antgroup/ditto-talkinghead .

# Download model checkpoints
RUN git lfs install && \
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints

# Build Cython extensions
RUN cd core/utils/blend && \
    python -m cython blend.pyx && \
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
        -I/usr/include/python3.10 \
        -o blend_impl.so blend_impl.c

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# Command to run inference (can be overridden)
CMD ["python", "inference.py", \
     "--data_root", "./checkpoints/ditto_trt_Ampere_Plus", \
     "--cfg_pkl", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", \
     "--audio_path", "./example/audio.wav", \
     "--source_path", "./example/image.png", \
     "--output_path", "./output/result.mp4"]