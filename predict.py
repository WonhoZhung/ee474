import os
import torch
import torch.nn as nn
import torch.utils.data
from model import SegNet
import torchvision
from PIL import Image
from pathlib import Path

def predict_image(dir):
    use_gpu = torch.cuda.is_available()
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    image_to_tensor = torchvision.transforms.ToTensor()
    tensor_to_image = torchvision.transforms.ToPILImage()

    save_path = Path(dir).parent

    image = Image.open(dir).convert('RGB')
    image = image_to_tensor(image)
    c, w, h = image.shape
    image = torch.reshape(image, (1, c, w, h))

    model = SegNet(3)
    if use_gpu:
        model = model.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
    model.load_state_dict(torch.load('./models/model.pth', map_location=device))

    model.eval()

    predicted = model(image)
    predicted[predicted > 1.0] = 1.0

    binary_predicted = predicted.clone()
    binary_mask = predicted.clone()
    binary_predicted[binary_predicted > 0.0] = 1.0
    binary_mask[binary_mask > 0.1] = 1.0
    masked = image + binary_mask
    masked[masked > 1.0] = 1.0

    predicted = predicted.squeeze().cpu()
    masked = masked.squeeze().cpu()
    image = tensor_to_image(predicted)
    image2 = tensor_to_image(masked)
    image.save(os.path.join(save_path, 'tmp_text.png'))
    image2.save(os.path.join(save_path, 'tmp_masked.png'))
