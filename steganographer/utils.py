import torchvision
from PIL import Image
import numpy as np


IMG_SIZE = 64
TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
    torchvision.transforms.ToTensor(),
])



def get_image_tensor_from_filepath(image_path, device):
    image = Image.open(image_path)
    return TRANSFORM(image).to(device)


def save_image_from_tensor_to_path(image_tensor, image_path):
    image_tensor.permute(1, 2, 0).numpy()
    image = Image.fromarray(np.uint8(image_tensor))
    image.save(image_path)
