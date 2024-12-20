import os
import cv2
from PIL import Image
import cairosvg
import subprocess


def svg_to_png(svg_file, png_file, width, height):
    """Convert SVG to PNG with specified width and height."""
    cairosvg.svg2png(
        url=svg_file, write_to=png_file, output_width=width, output_height=height
    )


def create_video_from_svgs(input_folder, output_video, fps=30, resolution=(1920, 1080)):
    """Create an MP4 video from SVG files in the folder with specified resolution."""
    # Step 1: Read and sort SVG files numerically by extracting the step number
    svg_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if f.startswith("step_") and f.endswith(".svg")
        ],
        key=lambda x: int(
            x.split("_")[1].split(".")[0]
        ),  # Sort by numeric part of file name
    )

    if not svg_files:
        print("No SVG files found in the folder.")
        return

    # Step 2: Convert SVGs to PNGs and get frame size
    temp_folder = os.path.join(input_folder, "temp_pngs")
    os.makedirs(temp_folder, exist_ok=True)
    png_files = []

    print("Converting SVG to PNG...")
    i = 0
    for svg_file in svg_files:
        print(f"Converting {svg_file} to PNG... {i+1}/{len(svg_files)}", end="\r")
        svg_path = os.path.join(input_folder, svg_file)
        png_path = os.path.join(temp_folder, f"{os.path.splitext(svg_file)[0]}.png")
        svg_to_png(svg_path, png_path, width=resolution[0], height=resolution[1])
        png_files.append(png_path)
        i += 1

    # Step 3: Get frame size from the first image
    sample_image = Image.open(png_files[0])
    width, height = sample_image.size
    frame_size = (width, height)

    # Step 4: Write video
    print("Creating video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for png_file in png_files:
        frame = cv2.imread(png_file)
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video}")

    # Step 5: Clean up temporary PNGs
    for png_file in png_files:
        os.remove(png_file)
    os.rmdir(temp_folder)
    print("Temporary files cleaned up.")


def convert_to_webm_and_ogg(intermediate_video, output_webm, output_ogg):
    """Convert intermediate video to WebM and OGG formats using FFmpeg."""
    print("Converting to high-quality WebM...")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            intermediate_video,
            "-c:v",
            "libvpx",
            "-b:v",
            "2M",
            "-crf",
            "10",
            "-c:a",
            "libvorbis",
            output_webm,
        ],
        check=True,
    )
    print(f"High-quality WebM video saved to {output_webm}")

    print("Converting to OGG...")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            intermediate_video,
            "-c:v",
            "libtheora",
            "-c:a",
            "libvorbis",
            output_ogg,
        ],
        check=True,
    )
    print(f"OGG video saved to {output_ogg}")


def compress_webm(
    input_webm, output_webm, target_bitrate="1.2M", crf=20, scale="1280:720"
):
    """Compress a WebM video to reduce file size."""
    print("Compressing WebM...")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_webm,
            "-c:v",
            "libvpx",
            "-b:v",
            target_bitrate,
            "-crf",
            str(crf),
            "-vf",
            f"scale={scale}",
            "-c:a",
            "libvorbis",
            output_webm,
        ],
        check=True,
    )
    print(f"Compressed WebM saved to {output_webm}")


def compress_mp4(
    input_file,
    output_file,
    target_bitrate="1M",
    crf=23,
    scale="1280:720",
    audio_bitrate="128k",
):
    """Compress an MP4 video to reduce file size."""
    print("Compressing MP4...")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_file,
            "-c:v",
            "libx264",
            "-b:v",
            target_bitrate,
            "-crf",
            str(crf),
            "-vf",
            f"scale={scale}",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            output_file,
        ],
        check=True,
    )
    print(f"Compressed MP4 saved to {output_file}")


# Input folder and output video path
ROOT_PATH = "./results/"

FOLDER_NAME_DEFAULT = "20241218-212551"
IMAGE_FOLDER_NAME = "images"
input_folder = os.path.join(
    ROOT_PATH, FOLDER_NAME_DEFAULT, IMAGE_FOLDER_NAME
)  # Folder containing SVG files

output_video = "output_video.mp4"  # Desired output video file name
intermediate_video = output_video  # Temporary MP4 file
output_webm = "output_video.webm"  # Final WebM video
output_ogg = "output_video.ogv"  # Final OGG video
compressed_mp4 = "output_compressed.mp4"  # Compressed
# Create video
# print("Creating video from SVG files...")
# create_video_from_svgs(input_folder, output_video, fps=20, resolution=(1280, 720))
# print("Video creation complete.")
# convert_to_webm_and_ogg(intermediate_video, output_webm, output_ogg)
# compress_webm(
#     "output_video.webm",
#     "output_compressed.webm",
#     target_bitrate="1.2M",
#     crf=20,
#     scale="1280:720",
# )
compress_mp4(
    output_video,
    compressed_mp4,
    target_bitrate="1M",
    crf=23,
    scale="720:360",
    audio_bitrate="128k",
)
