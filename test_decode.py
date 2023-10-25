import sys
import numpy as np
import torch
from PIL import Image
from decoder import Decoder
from utils import device
from arithmetic_compressor.models import SimpleAdaptiveModel
from arithmetic_compressor import AECompressor


def decode(B, weights_path, encoded, img_path):
    sys.set_int_max_str_digits(1000000)
    model = Decoder(B=B)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()

    [res, size] = encoded.split(",")
    mask = list(map(int, list(str(bin(int(res)))[3:])))
    encoded = np.asarray(arithmetic_decode(mask, int(size), B))

    encoded_tensor = torch.from_numpy(dequantize(encoded, B)).unsqueeze(0).float()
    answer = torch.sigmoid(model(encoded_tensor))
    answer *= 255
    img = Image.fromarray((torch.permute(answer, (0, 2, 3, 1)).detach().cpu().numpy()[0]).astype(np.uint8))
    img.save(img_path)
    return answer


def dequantize(encoded, B):
    return (encoded / (2 ** B)).astype(float)


def arithmetic_decode(res, size, B):
    keys = [key for key in range(0, 2 ** B + 1)]
    prob = 1 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)

    return coder.decompress(res, size)


if __name__ == '__main__':
    B = int(sys.argv[1])
    weights_path = sys.argv[2]
    txt_path = sys.argv[3]
    img_path = sys.argv[4]
    with open(txt_path, 'r') as file:
        encoded = file.read()
    result = decode(B, weights_path, encoded, img_path)
