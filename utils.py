import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt


def image_loader(image_path, image_size):
    image = Image.open(image_path)
    image_transforms = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)]
    )
    image = image_transforms(image).unsqueeze(0)
    return image


def gram_matrix(input_tensor):
    batch_size, channel_size, height, width = input_tensor.size()
    features = input_tensor.view(batch_size * channel_size, height * width)
    G = torch.mm(features, features.t())

    return G.div(batch_size * channel_size * height * width)


def next_path(path_pattern, index=1):
    while os.path.exists(path_pattern % index):
        index += 1
    return path_pattern % index


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)

    image_transform = transforms.ToPILImage()
    image = image_transform(image)

    if title is not None:
        plt.title(title)

    plt.imshow(image)
    plt.pause(0.001)
