from autoencoder import Autoencoder
import torch
from utils import device, transform
from PIL import Image
import sys
import numpy as np


def autoencode(B, weights_path, image_path, res_path):
    model = Autoencoder(B=B)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    img = transform(Image.open(image_path)).unsqueeze(0)
    answer = model(img)
    answer *= 255
    img = Image.fromarray((torch.permute(answer, (0, 2, 3, 1)).detach().cpu().numpy()[0]).astype(np.uint8))
    img.save(res_path)
    return img


if __name__ == '__main__':
    B = int(sys.argv[1])
    weights_path = sys.argv[2]
    image_path = sys.argv[3]
    res_path = sys.argv[4]
    result = autoencode(B, weights_path, image_path, res_path)
