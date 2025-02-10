docker run --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints ditto-talkinghead \
  python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "/app/input/your_audio.wav" \
    --source_path "/app/input/your_image.png" \
    --output_path "/app/output/result.mp4"