import os
from PIL import Image
from torchvision import transforms

def load_image(image_path, device, output_size=None):
    """Loads an image by transforming it into a tensor."""
    img = Image.open(image_path)

    output_dim = None
    if output_size is None:
        output_dim = (img.size[1], img.size[0])
    elif isinstance(output_size, int):
        output_dim = (output_size, output_size)
    elif isinstance(output_size, tuple):
        if (len(output_size) == 2) and isinstance(output_size[0], int) and isinstance(output_size[1], int):
            output_dim = output_size
    else:
        raise ValueError("ERROR: output_size must be an integer or a 2-tuple of (height, width) if provided.")

    torch_loader = transforms.Compose(
        [
            transforms.Resize(output_dim),
            transforms.ToTensor()
        ]
    )
    
    img_tensor = torch_loader(img).unsqueeze(0)
    return img_tensor.to(device)


def get_image_name_ext(img_path):
    """Get name and extension of the image file from its path."""
    return os.path.splitext(os.path.basename(img_path))[0], os.path.splitext(os.path.basename(img_path))[1][1:]
