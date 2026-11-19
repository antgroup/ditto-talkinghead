import imageio
import os


class VideoWriterByImageIO:
    def __init__(self, video_path, fps=25, **kwargs):
        video_format = kwargs.get("format", "mp4")  # default is mp4 format
        codec = kwargs.get("vcodec", "libx264")  # default is libx264 encoding
        quality = kwargs.get("quality")  # video quality
        pixelformat = kwargs.get("pixelformat", "yuv420p")  # video pixel format
        macro_block_size = kwargs.get("macro_block_size", 2)
        ffmpeg_params = ["-crf", str(kwargs.get("crf", 18))]

        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            format=video_format,
            codec=codec,
            quality=quality,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixelformat,
            macro_block_size=macro_block_size,
        )
        self.writer = writer

    def __call__(self, img, fmt="bgr"):
        if fmt == "bgr":
            frame = img[..., ::-1]
        else:
            frame = img
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()
