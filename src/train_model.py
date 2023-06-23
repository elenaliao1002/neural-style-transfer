import os
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.utils import save_image


class ImageStyleTransfer_VGG19(nn.Module):
    def __init__(self):
        super(ImageStyleTransfer_VGG19, self).__init__()

        self.chosen_features = {0: 'conv11', 5: 'conv21', 10: 'conv31', 19: 'conv41', 28: 'conv51'}
        self.model = vgg19(weights='DEFAULT').features[:29]

    def forward(self, x):
        feature_maps = dict()
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.chosen_features.keys():
                feature_maps[self.chosen_features[idx]] = x
        
        return feature_maps


def _get_content_loss(content_feature, generated_feature):
    """Compute MSE between content feature map and generated feature map as content loss."""
    return torch.mean((generated_feature - content_feature) ** 2)


def _get_style_loss(style_feature, generated_feature):
    """Compute MSE between gram matrix of style feature map and of generated feature map as style loss."""
    _, channel, height, width = generated_feature.shape
    style_gram = style_feature.view(channel, height*width).mm(
        style_feature.view(channel, height*width).t()
    )
    generated_gram = generated_feature.view(channel, height*width).mm(
        generated_feature.view(channel, height*width).t()
    )

    return torch.mean((generated_gram - style_gram) ** 2)


def train_image(content, style, generated, device, train_config, output_dir, output_img_fmt, content_img_name, style_img_name, verbose=False):
    """Update the output image using pre-trained VGG19 model."""
    model = ImageStyleTransfer_VGG19().to(device).eval()    # freeze parameters in the model

    # set default value for each configuration if not specified in train_config
    num_epochs = train_config.get('num_epochs') if train_config.get('num_epochs') is not None else 6000
    lr = train_config.get('learning_rate') if train_config.get('learning_rate') is not None else 0.001
    alpha = train_config.get('alpha') if train_config.get('alpha') is not None else 1
    beta = train_config.get('beta') if train_config.get('beta') is not None else 0.01
    capture_content_features_from = train_config.get('capture_content_features_from') \
        if train_config.get('capture_content_features_from') is not None else {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
    capture_style_features_from = train_config.get('capture_style_features_from') \
        if train_config.get('capture_style_features_from') is not None else {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
            
    # check if values passed to capture_content_features_from and capture_style_features_from are valid
    if not isinstance(capture_content_features_from, set):
        if isinstance(capture_content_features_from, dict):
            capture_content_features_from = set(capture_content_features_from.keys())
        elif isinstance(capture_content_features_from, str):
            capture_content_features_from = set([item.strip() for item in capture_content_features_from.split(',')])
        else:
            print(f"ERROR: invalid value for 'capture_content_features_from' in training configuration file: {capture_content_features_from}.")
            return 0
        
    if not capture_content_features_from.issubset({'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}):
        print(f"ERROR: invalid value for 'capture_content_features_from' in training configuration file: {capture_content_features_from}.")
        return 0
    
    if not isinstance(capture_style_features_from, set):
        if isinstance(capture_style_features_from, dict):
            capture_style_features_from = set(capture_style_features_from.keys())
        elif isinstance(capture_style_features_from, str):
            capture_style_features_from = set([item.strip() for item in capture_style_features_from.split(',')])
        else:
            print(f"ERROR: invalid value for 'capture_style_features_from' in training configuration file: {capture_style_features_from}.")
            return 0
        
    if not capture_style_features_from.issubset({'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}):
        print(f"ERROR: invalid value for 'capture_style_features_from' in training configuration file: {capture_style_features_from}.")
        return 0

    optimizer = torch.optim.Adam([generated], lr=lr)

    if verbose:
        # create a directory to save intermediate results
        intermediate_dir = os.path.join(output_dir, f'nst-{content_img_name}-{style_img_name}-intermediate')
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)

    for epoch in range(num_epochs):
        # get features maps of content, style and generated images from chosen layers
        content_features = model(content)
        style_features = model(style)
        generated_features = model(generated)

        content_loss = style_loss = 0

        for layer_name in generated_features.keys():
            content_feature = content_features[layer_name]
            style_feature = style_features[layer_name]
            generated_feature = generated_features[layer_name]

            if layer_name in capture_content_features_from:
                content_loss_per_feature = _get_content_loss(content_feature, generated_feature)
                content_loss += content_loss_per_feature
            
            if layer_name in capture_style_features_from:
                style_loss_per_feature = _get_style_loss(style_feature, generated_feature)
                style_loss += style_loss_per_feature

        # compute loss 
        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print loss value and save progress every 200 epochs
        if verbose:
            if (epoch + 1) % 200 == 0:
                save_image(generated, os.path.join(intermediate_dir, f'nst-{content_img_name}-{style_img_name}-{epoch + 1}.{output_img_fmt}'))

                print(f"\tEpoch {epoch + 1}/{num_epochs}, loss = {total_loss.item()}")
    
    if verbose:
        print("\t================================")
        print(f"\tIntermediate images are saved in directory: '{intermediate_dir}'")
        print("\t================================")

    return 1


def train_frame(content, style, generated, device, output_img_fmt):
    """Update the output image using pre-trained VGG19 model for video transfer."""
    model = ImageStyleTransfer_VGG19().to(device).eval()    # freeze parameters in the model

    # set default value for each configuration
    num_epochs = 2000
    lr = 0.01
    alpha = 50
    beta = 0.001
    capture_content_features_from = {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}
    capture_style_features_from = {'conv11', 'conv21', 'conv31', 'conv41', 'conv51'}

    optimizer = torch.optim.Adam([generated], lr=lr)

    for epoch in range(num_epochs):
        # get features maps of content, style and generated images from chosen layers
        content_features = model(content)
        style_features = model(style)
        generated_features = model(generated)

        content_loss = style_loss = 0

        for layer_name in generated_features.keys():
            content_feature = content_features[layer_name]
            style_feature = style_features[layer_name]
            generated_feature = generated_features[layer_name]
            
            if layer_name in capture_content_features_from:
                content_loss_per_feature = _get_content_loss(content_feature, generated_feature)
                content_loss += content_loss_per_feature

            if layer_name in capture_style_features_from:
                style_loss_per_feature = _get_style_loss(style_feature, generated_feature)
                style_loss += style_loss_per_feature

        # compute loss
        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return 1
