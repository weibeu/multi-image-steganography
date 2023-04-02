from torchvision import transforms
from PIL import Image
import numpy as np


def get_image_tensor_from_filepath(image_path, device):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(image).to(device)


def save_image_from_tensor_to_path(image_tensor, image_path):
    image_tensor.permute(1, 2, 0).numpy()
    image = Image.fromarray(np.uint8(image_tensor))
    image.save(image_path)
