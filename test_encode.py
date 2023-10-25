from encoder import Encoder
import torch
from utils import device, transform
from PIL import Image
import sys
from arithmetic_compressor.models import SimpleAdaptiveModel
from arithmetic_compressor import AECompressor


def encode(B, weights_path, image_path, res_path):
    sys.set_int_max_str_digits(1000000)
    model = Encoder(B=B)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    img = transform(Image.open(image_path)).unsqueeze(0)
    answer = model(img).detach().cpu().numpy()[0]
    answer = quantize(answer, B)

    res, size = arithmetic_encode(answer, B)
    answer = str(int("".join(map(str, [1] + res)), 2)) + "," + str(size)
    with open(res_path, 'w') as file:
        file.write(answer)
    return answer


def quantize(answer, B):
    return (answer * (2 ** B) + 0.5).astype(int)


def arithmetic_encode(answer, B):
    keys = [key for key in range(0, 2 ** B + 1)]
    prob = 1 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)

    return coder.compress(answer), len(answer)


if __name__ == '__main__':
    B = int(sys.argv[1])
    weights_path = sys.argv[2]
    image_path = sys.argv[3]
    res_path = sys.argv[4]
    result = encode(B, weights_path, image_path, res_path)
