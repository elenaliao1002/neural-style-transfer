import argparse
import os
import sys
import PIL
import yaml
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image

from src.process_image import load_image, get_image_name_ext
from src.train_model import train_image


def image_style_transfer(config):
    """Implements neural style transfer on a content image using a style image, applying provided configuration."""
    if config.get('image_dir') is not None:
        image_dir = config.get('image_dir')
        content_path = os.path.join(image_dir, config.get('content_filename'))
        style_path = os.path.join(image_dir, config.get('style_filename'))
        output_dir = config.get('output_dir') if config.get('output_dir') is not None else image_dir
    else:
        output_dir = config.get('output_dir')
        content_path = config.get('content_filepath')
        style_path = config.get('style_path')

    verbose = not config.get('quiet')

    if verbose:
        print("Loading content and style images...")
    
    try:
        content_img = Image.open(content_path)
    except FileNotFoundError:
        print(f"ERROR: could not find such file: '{content_path}'.")
        return
    except PIL.UnidentifiedImageError:
        print(f"ERROR: could not identify image file: '{content_path}'.")
        return

    try:
        style_img = Image.open(style_path)
    except FileNotFoundError:
        print(f"ERROR: could not find such file: '{style_path}'.")
        return
    except PIL.UnidentifiedImageError:
        print(f"ERROR: could not identify image file: '{style_path}'.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load content and style images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = config.get('output_image_size')
    if output_size is not None:
        if len(output_size) > 1: 
            output_size = tuple(output_size)
        else:
            output_size = output_size[0]

    content_tensor = load_image(content_path, device, output_size=output_size)
    output_size = (content_tensor.shape[2], content_tensor.shape[3])
    style_tensor = load_image(style_path, device, output_size=output_size)

    if verbose:
        print("Content and style images successfully loaded.")
        print()
        print("Initializing output image...")

    # initialize output image
    generated_tensor = content_tensor.clone().requires_grad_(True)

    if verbose:
        print("Output image successfully initialized.")
        print()

    # load training configuration if provided
    train_config = dict()
    if (train_config_path := config.get('train_config_path')) is not None:
        if verbose:
            print("Loading training configuration file...")

        try:
            with open(train_config_path, 'r') as f:
                train_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"ERROR: could not find such file: '{train_config_path}'.")
            return
        except yaml.YAMLError:
            print(f"ERROR: fail to load yaml file: '{train_config_path}'.")
            return

        if verbose:
            print("Training configuration file successfully loaded.")
            print()
        
    if verbose:
        print("Training...")
    
    content_img_name, content_img_fmt = get_image_name_ext(content_path)
    style_img_name, _ = get_image_name_ext(style_path)

    output_img_fmt = config.get('output_image_format')
    if output_img_fmt == 'same':
        output_img_fmt = content_img_fmt

    # train model
    success = train_image(content_tensor, style_tensor, generated_tensor, device, train_config, output_dir, output_img_fmt, content_img_name, style_img_name, verbose=verbose)

    # save output image to specified directory
    if success:
        save_image(generated_tensor, os.path.join(output_dir, f'nst-{content_img_name}-{style_img_name}-final.{output_img_fmt}'))

    if verbose:
        print(f"Output image successfully generated as {os.path.join(output_dir, f'nst-{content_img_name}-{style_img_name}-final.{output_img_fmt}')}.")


def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Path to the directory where content image and style image are stored.")
    parser.add_argument("--content_filename", type=str, default="content.jpg", help="File name of the content image in image_dir. Will use \"content.jpg\" if not provided.")
    parser.add_argument("--style_filename", type=str, default="style.jpg", help="File name of the style image in image_dir. Will use \"style.jpg\" if not provided.")
    parser.add_argument("--content_filepath", required="--image_dir" not in sys.argv, type=str, help="Path to the content image if image_dir not provided.")
    parser.add_argument("--style_filepath", required="--image_dir" not in sys.argv, type=str, help="Path to the style image if image_dir not provided.")
    parser.add_argument("--output_dir", required="--image_dir" not in sys.argv, type=str, help="Directory that stores the output image. Will be the same as image_dir if not provided while image_dir provided.")
    parser.add_argument("--output_image_size", nargs="+", type=int, help="Size of the output image. Either one integer or two integers separated by space is accepted. Will use the dimensions of content image if not provided.")
    parser.add_argument("--output_image_format", choices=["jpg", "png", "jpeg", "same"], default="jpg", help="Format of the output image. Can be either \"jpg\", \"png\", \"jpeg\", or \"same\". If \"same\", output image will have the same format as the content image. \"jpg\" will be the default format.")
    parser.add_argument("--train_config_path", type=str, help="Path to training configuration file in .yaml format. May include: num_epochs, learning_rate, alpha, beta, capture_content_features_from, capture_style_features_from.")
    parser.add_argument("--quiet", type=bool, default=False, help="True stops showing debugging messages, loss function values during training process, and stops generating intermediate images.")

    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    
    image_style_transfer(config)


if __name__ == '__main__':
    main()
    