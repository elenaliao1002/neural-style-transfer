import cv2
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str, help="Path to the image.")
    
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
        
    image_path = config['image_path']
    
    try:
        height, width, _ = cv2.imread(image_path).shape
    except AttributeError:
        print(f'Error: Invalid image path: {image_path}.')
        exit(1)
    
    print(f'Image height: {height}, width: {width}.')