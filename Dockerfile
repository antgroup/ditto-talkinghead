FROM nvcr.io/nvidia/tensorrt:23.12-py3

# Rest of the Dockerfile remains the same...
# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with --no-deps for torch to avoid CUDA dependencies
RUN pip install --no-cache-dir \
    torch --no-deps \
    && pip install --no-cache-dir \
    cuda-python \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv-python-headless \
    scikit-image \
    cython \
    imageio-ffmpeg \
    colored \
    numpy==2.0.1

# Clone the repository (without checkpoints)
RUN git clone https://github.com/antgroup/ditto-talkinghead .

# Build Cython extensions
RUN cd core/utils/blend && \
    cython blend.pyx && \
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
        -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
        -I$(python3 -c "import numpy; print(numpy.get_include())") \
        blend.c -o blend_impl.so

# Set environment variables
ENV PYTHONPATH=/app

# Command to run inference (can be overridden)
CMD ["python", "inference.py", \
     "--data_root", "./checkpoints/ditto_trt_Ampere_Plus", \
     "--cfg_pkl", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", \
     "--audio_path", "./example/audio.wav", \
     "--source_path", "./example/image.png", \
     "--output_path", "./output/result.mp4"]