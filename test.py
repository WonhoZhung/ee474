import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import DnCNN
from model import SegNet
from torch.optim.lr_scheduler import MultiStepLR
from utils import AverageMeter
from dataset import get_dataloader
import torchvision
from PIL import Image

use_gpu = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
batch_size = 16
data_dir = './data/garfield_dataset'

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((172, 600)),
    torchvision.transforms.ToTensor()
])

dataloader = get_dataloader(data_dir, transform=transforms, batch_size=batch_size)


def test_autoencoder(epoch_plus):

    # model = autoencoder(nchannels=3, width=172, height=600)
    model = SegNet(3)
    if ngpu > 1:
        model = nn.DataParallel(model)
    if use_gpu:
        model = model.to(device, non_blocking=True)
    if epoch_plus > 0:
        model.load_state_dict(torch.load('./models/autoencoder_{}.pth'.format(epoch_plus)))

    model.eval()

    index = 1
    for data in dataloader:
        gt, text = data
        if use_gpu:
            gt, text = gt.to(device, non_blocking=True), text.to(device, non_blocking=True)

        predicted = model(text)

        save_path = './autoencoder_results_3'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        trans = torchvision.transforms.ToPILImage()
        for i in range(batch_size):
            image = trans(predicted[i])
            image.save(os.path.join(save_path, 'extracted_ ({}).png'.format(index)))
            index += 1

