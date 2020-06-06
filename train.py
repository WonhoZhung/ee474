import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import DnCNN
from model import SegNet
from model import autoencoder
from torch.optim.lr_scheduler import MultiStepLR
from utils import AverageMeter
from dataset import get_dataloader

use_gpu = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
batch_size = 16
data_dir = './data/garfield_dataset'

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((172, 600)),
    torchvision.transforms.ToTensor()
])

dataloader = get_dataloader(data_dir, train=True, transform=transforms, batch_size=batch_size)

def train_dncnn(epoch_plus):
    writer = SummaryWriter(log_dir='./runs')
    num_epochs = 100
    lr = 0.001

    model = DnCNN(num_layers=17)
    if ngpu > 1:
        model = nn.DataParallel(model)
    if use_gpu:
        model = model.to(device, non_blocking=True)
    if epoch_plus > 0:
        model.load_state_dict(torch.load('./models/DnCNN_model_{}_epochs.pth'.format(epoch_plus)))
    criterion = nn.MSELoss(size_average=False, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    for epoch in range(num_epochs):
        scheduler.step(epoch)
        epoch_losses = AverageMeter()

        with tqdm(total=(1000 - 1000 % batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1 + epoch_plus, num_epochs + epoch_plus))
            for data in dataloader:
                gt, text = data
                if use_gpu:
                    gt, text = gt.to(device, non_blocking=True), text.to(device, non_blocking=True)

                predicted = model(text)

                loss = criterion(predicted, gt) / (2 * len(gt))
                epoch_losses.update(loss.item(), len(gt))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(gt))

        save_path = './models'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'DnCNN_model_{}_epochs.pth'.format(epoch + 1 + epoch_plus)))
        writer.add_scalar('Loss', epoch_losses.avg, epoch + 1 + epoch_plus)
        writer.add_image('gt/gt_image_{}'.format(epoch + 1 + epoch_plus), gt[0].squeeze(), epoch + 1 + epoch_plus)
        writer.add_image('text/text_image_{}'.format(epoch + 1 + epoch_plus), text[0].squeeze(), epoch + 1 + epoch_plus)
        writer.add_image('predicted/predicted_image_{}'.format(epoch + 1 + epoch_plus), predicted[0].squeeze(), epoch + 1 + epoch_plus)

    writer.close()


def train_autoencoder(epoch_plus):
    writer = SummaryWriter(log_dir='./runs_autoencoder_3')
    num_epochs = 500
    lr = 0.001
    bta1 = 0.9
    bta2 = 0.999
    weight_decay = 0.001

    # model = autoencoder(nchannels=3, width=172, height=600)
    model = SegNet(3)
    if ngpu > 1:
        model = nn.DataParallel(model)
    if use_gpu:
        model = model.to(device, non_blocking=True)
    if epoch_plus > 0:
        model.load_state_dict(torch.load('./models/autoencoder_{}.pth'.format(epoch_plus)))
    criterion_bce = nn.BCELoss()
    # criterion_dice = nn.
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(bta1, bta2), weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(1000 - 1000 % batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1 + epoch_plus, num_epochs + epoch_plus))
            for data in dataloader:
                gt, text = data
                if use_gpu:
                    gt, text = gt.to(device, non_blocking=True), text.to(device, non_blocking=True)

                predicted = model(text)

                # loss = criterion_bce(predicted, gt) + criterion_dice(predicted, gt)
                loss = criterion(predicted, gt - text) # predicts extracted text in white, all others in black
                epoch_losses.update(loss.item(), len(gt))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(gt))

        save_path = './autoencoder_models_3'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        gt_text = gt - text
        predicted_mask = text + predicted

        torch.save(model.state_dict(),
                   os.path.join(save_path, 'autoencoder_{}.pth'.format(epoch + 1 + epoch_plus)))
        writer.add_scalar('Loss', epoch_losses.avg, epoch + 1 + epoch_plus)
        writer.add_image('text/text_image_{}'.format(epoch + 1 + epoch_plus), text[0].squeeze(), epoch + 1 + epoch_plus)
        writer.add_image('gt/gt_image_{}'.format(epoch + 1 + epoch_plus), gt[0].squeeze(), epoch + 1 + epoch_plus)
        writer.add_image('gt_text/gt_image_{}'.format(epoch + 1 + epoch_plus), gt_text[0].squeeze(), epoch + 1 + epoch_plus)
        writer.add_image('predicted/predicted_image_{}'.format(epoch + 1 + epoch_plus), predicted_mask[0].squeeze(),
                         epoch + 1 + epoch_plus)
        writer.add_image('predicted_text/predicted_image_{}'.format(epoch + 1 + epoch_plus), predicted[0].squeeze(),
                         epoch + 1 + epoch_plus)

    writer.close()
