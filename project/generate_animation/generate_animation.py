import os
import cv2
from PIL import Image
import cairosvg


def svg_to_png(svg_file, png_file):
    """Convert SVG to PNG."""
    cairosvg.svg2png(url=svg_file, write_to=png_file)


def create_video_from_svgs(input_folder, output_video, fps=30):
    """Create an MP4 video from SVG files in the folder."""
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
        svg_to_png(svg_path, png_path)
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


# Input folder and output video path
ROOT_PATH = "./results/"

FOLDER_NAME_DEFAULT = "20241217-101210"
IMAGE_FOLDER_NAME = "images"
input_folder = os.path.join(
    ROOT_PATH, FOLDER_NAME_DEFAULT, IMAGE_FOLDER_NAME
)  # Folder containing SVG files

output_video = "output_video.mp4"  # Desired output video file name

# Create video
print("Creating video from SVG files...")
create_video_from_svgs(input_folder, output_video, fps=30)
print("Video creation complete.")
