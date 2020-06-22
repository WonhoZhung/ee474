import os
import torch
import torch.nn as nn
import torch.utils.data
from model import SegNet
import torchvision
from PIL import Image


def test_autoencoder(epoch_plus, text, index):
    use_gpu = torch.cuda.is_available()
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = SegNet(3)
    if ngpu > 1:
        model = nn.DataParallel(model)
    if use_gpu:
        model = model.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
    if epoch_plus > 0:
        model.load_state_dict(torch.load('./autoencoder_models_2/autoencoder_{}.pth'.format(epoch_plus)))

    model.eval()

    if use_gpu:
        text.to(device, non_blocking=True)

    predicted = model(text)
    predicted[predicted > 1.0] = 1.0

    save_path1 = './results/text'
    save_path2 = './results/masked'
    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
    if not os.path.exists(save_path2):
        os.mkdir(save_path2)

    binary_predicted = predicted.clone()
    binary_mask = predicted.clone()
    binary_predicted[binary_predicted > 0.0] = 1.0
    binary_mask[binary_mask > 0.1] = 1.0
    masked = text + binary_mask
    masked[masked > 1.0] = 1.0

    trans = torchvision.transforms.ToPILImage()

    predicted = predicted.squeeze().cpu()
    masked = masked.squeeze().cpu()
    image = trans(predicted)
    image2 = trans(masked)
    image.save(os.path.join(save_path1, 'text_{}.png'.format(index)))
    image2.save(os.path.join(save_path2, 'masked_{}.png'.format(index)))
    del text
    del predicted
    del masked
    del binary_predicted


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

for index in range(1, 19):
    image_path = './data/test_images/{}.png'.format(index)
    image = Image.open(image_path).convert('RGB')
    image = transforms(image)
    c, w, h = image.shape
    image = torch.reshape(image, (1, c, w, h))
    test_autoencoder(epoch_plus=88, text=image, index=index)

# for index in range(1, 400):
#     image_path = './data/test_images/{}.png'.format(16)
#     image = Image.open(image_path).convert('RGB')
#     image = transforms(image)
#     c, w, h = image.shape
#     image = torch.reshape(image, (1, c, w, h))
#     test_autoencoder(epoch_plus=index, text=image, index=index)


