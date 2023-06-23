import argparse
import os
import sys
import PIL
import yaml
import cv2
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image

from src.process_image import load_image, get_image_name_ext
from src.train_model import train_frame


def _image_style_transfer(content_frame_path, style_path, output_frame_path, output_size):
    try:
        content_img = Image.open(content_frame_path)
    except FileNotFoundError:
        print(f"ERROR: could not find such file: '{content_frame_path}'.")
        return
    except PIL.UnidentifiedImageError:
        print(f"ERROR: could not identify image file: '{content_frame_path}'.")
        return

    try:
        style_img = Image.open(style_path)
    except FileNotFoundError:
        print(f"ERROR: could not find such file: '{style_path}'.")
        return
    except PIL.UnidentifiedImageError:
        print(f"ERROR: could not identify image file: '{style_path}'.")
        return

    # load content and style images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_tensor = load_image(content_frame_path, device, output_size=output_size)
    output_size = (content_tensor.shape[2], content_tensor.shape[3])
    style_tensor = load_image(style_path, device, output_size=output_size)

    # initialize output image
    generated_tensor = content_tensor.clone().requires_grad_(True)

    output_img_fmt = 'jpg'
    
    # train model
    success = train_frame(content_tensor, style_tensor, generated_tensor, device, output_img_fmt)

    # save output image to specified directory
    if success:
        save_image(generated_tensor, output_frame_path)

    return success

def video_style_transfer(config):
    """Implements neural style transfer on a video using a style image, applying provided configuration."""
    if config.get('file_dir') is not None:
        file_dir = config.get('file_dir')
        content_video_path = os.path.join(file_dir, config.get('content_filename'))
        style_path = os.path.join(file_dir, config.get('style_filename'))
        output_dir = config.get('output_dir') if config.get('output_dir') is not None else file_dir
    else:
        output_dir = config.get('output_dir')
        content_video_path = config.get('content_filepath')
        style_path = config.get('style_path')
    
    output_size = config.get('output_frame_size')
    if output_size is not None:
        if len(output_size) > 1: 
            output_size = tuple(output_size)
        else:
            output_size = output_size[0]
    
    verbose = not config.get('quiet')

    if not os.path.exists(os.path.join(output_dir, "content_frames")):
        os.makedirs(os.path.join(output_dir, "content_frames"))

    if verbose:
        print("Loading content video...")

    cap = cv2.VideoCapture(content_video_path)
    # retrieve metadata from content video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    content_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print(f"ERROR: could not retrieve frames from content video at path: '{content_video_path}'.")
        return

    # extract frames from content video
    for i in range(total_frames):
        success, img = cap.read()
        if success:
            cv2.imwrite(os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg"), img)
        else:
            print(F'ERROR: {os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")} failed to be extracted.')
            return

    cap.release()

    if verbose:
        print("Frames successfully extracted from content video.")
        print()
        print("Performing image style transfer for each frame...")

    if not os.path.exists(os.path.join(output_dir, "transferred_frames")):
        os.makedirs(os.path.join(output_dir, "transferred_frames"))

    # perform image style transfer with each content frame and style image
    for i in range(total_frames):
        content_frame_path = os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")
        output_frame_path = os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg")
        success = _image_style_transfer(content_frame_path, style_path, output_frame_path, output_size)

        if verbose:
            if success:
                print(f'\tImage style transfer success for frame {content_frame_path}.')
            else:
                print(f'\tWarning: Image style transfer failed for frame {content_frame_path}.')
                return
    
    if verbose:
        print("Image style transfer complete.")
        print()
        print("Synthesizing video from transferred frames...")
    
    content_video_name, _ = get_image_name_ext(content_video_path)
    style_img_name, _ = get_image_name_ext(style_path)
    output_video_path = os.path.join(output_dir, f"nst-{content_video_name}-{style_img_name}-final.mp4")

    output_frame_height, output_frame_width, _ = cv2.imread(os.path.join(output_dir, "transferred_frames", "transferred_frame-00000001.jpg")).shape
    output_fps = config.get('fps') if config.get('fps') is not None else content_fps
    # synthesize video using transferred content frames
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, cv2_fourcc, output_fps, (output_frame_width, output_frame_height), True)

    for i in range(total_frames):
        frame = cv2.imread(os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg"))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()

    if verbose:
        print(f'Video successfully synthesized to {output_video_path}.')


def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, help="Path to the directory where content video and style image are stored.")
    parser.add_argument("--content_filename", type=str, default="content.mp4", help="File name of the content video in file_dir. Will use \"content.mp4\" if not provided.")
    parser.add_argument("--style_filename", type=str, default="style.jpg", help="File name of the style image in image_dir. Will use \"style.jpg\" if not provided.")
    parser.add_argument("--content_filepath", required="--file_dir" not in sys.argv, type=str, help="Path to the content video if file_dir not provided.")
    parser.add_argument("--style_filepath", required="--file_dir" not in sys.argv, type=str, help="Path to the style image if image_dir not provided.")
    parser.add_argument("--output_dir", required="--file_dir" not in sys.argv, type=str, help="Directory that stores the output video. Will be the same as file_dir if not provided while image_dir provided.")
    parser.add_argument("--output_frame_size", nargs="+", type=int, help="Size of frames of output video. Either one integer or two integers (height, weight) separated by space is accepted. Will use the dimensions of frames of content video if not provided.")
    parser.add_argument("--fps", type=int, help="FPS of output video. Will use the FPS of content video if not provided.")
    parser.add_argument("--quiet", type=bool, default=False, help="True stops showing debugging messages, loss function values during training process, and stops generating intermediate images.")

    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    video_style_transfer(config)


if __name__ == '__main__':
    main()