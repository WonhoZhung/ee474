import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

os.chdir('/home/sgvr/wkim97/EE474/project')

class dataset(Dataset):
    """Customized `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    dataset
        `-- data
            |-- train_data
                |-- gt_0.jpg
                |-- ...
                `-- gt_10084.jpg

                |-- text_0.jpg
                |-- ...
                `-- text_10084.jpg

    """

    def __init__(self, root, train=True, transform=None):
        super(dataset, self).__init__()
        """
        Instructions: 
            1. Assume that `root` equals to `dataset/cifar10`.

            2. If `train` is True, then parse all paths of train images, and keep them in the list `self.paths`. 
               E.g.) self.paths = ['dataset/cifar10/train/0/00001.png', ..., 'dataset/cifar10/train/9/4800.png']
               Also, the length of `self.paths` list should be 48,000.

            3. If `train` is False, then parse all paths of test images, and keep them in the list `self.paths`. 
               E.g.) self.paths = ['dataset/cifar10/test/0/04801.png', ..., 'dataset/cifar10/test/9/06000.png']
               Also, the length of `self.paths` list should be 12,000.

        Args:
            root (string): Root directory of dataset where directory ``cifar10`` exists.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set. (default: True)
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop`` (default: None)
        """
        self.transform = transform

        ################################
        self.gt_paths = []
        self.text_paths = []
        path = root
        if train == True:
            path = os.path.join(path, 'train')
            for j in range(1000):
                gt_tmp = path + '/masked/masked_ ({}).png'.format(j + 1)
                self.gt_paths.append(gt_tmp)
                text_tmp = path + '/text/text_ ({}).png'.format(j + 1)
                self.text_paths.append(text_tmp)

            ################################

            assert isinstance(self.gt_paths, (list,)), 'Wrong type. self.paths should be list.'
            assert isinstance(self.text_paths, (list,)), 'Wrong type. self.paths should be list.'
            assert len(
                self.gt_paths) == 1000, 'There are 1000 train images, but you have gathered %d image paths' % len(
                self.gt_paths)
            assert len(
                self.text_paths) == 1000, 'There are 1000 test images, but you have gathered %d image paths' % len(
                self.text_paths)
        else:
            path = os.path.join(path, 'test')
            for j in range(100):
                gt_tmp = path + '/masked/masked_ ({}).png'.format(j + 1)
                self.gt_paths.append(gt_tmp)
                text_tmp = path + '/text/text_ ({}).png'.format(j + 1)
                self.text_paths.append(text_tmp)

            ################################

            assert isinstance(self.gt_paths, (list,)), 'Wrong type. self.paths should be list.'
            assert isinstance(self.text_paths, (list,)), 'Wrong type. self.paths should be list.'
            assert len(
                self.gt_paths) == 100, 'There are 1000 train images, but you have gathered %d image paths' % len(
                self.gt_paths)
            assert len(
                self.text_paths) == 100, 'There are 1000 test images, but you have gathered %d image paths' % len(
                self.text_paths)



    def __getitem__(self, idx):
        """
        Instructions:
            1. Given a path of an image, which is grabbed by self.paths[idx], infer the class label of the image.
            2. Convert the inferred class label into torch.LongTensor with shape (), and keep it in `label` variable.`

        Args:
            idx (int): Index of self.paths

        Returns:
            image (torch.FloatTensor): An image tensor of shape (3, 32, 32).
            label (torch.LongTensor): A label tensor of shape ().
        """

        gt = self.gt_paths[idx]
        # P4.2. Infer class label from `path`,
        # write your code here.
        text = self.text_paths[idx]

        # P4.3. Convert it to torch.LongTensor with shape ().
        gt = Image.open(gt).convert('RGB')
        if self.transform is not None:
            gt = self.transform(gt)
        text = Image.open(text).convert('RGB')
        if self.transform is not None:
            text = self.transform(text)

        return gt, text

    def __len__(self):
        return len(self.gt_paths)

# # Check and test your CIFAR10 Dataset class here.
# data_dir = 'data/train_data'
# train = True
# transform = transforms.ToTensor()
#
# dset = dataset(data_dir, train, transform)
# print('num data:', len(dset))
#
# x_test, y_test = dset[0]
# print('gt shape:', x_test.shape, '| type:', x_test.dtype)
# print('text shape:', y_test.shape, '| type:', y_test.dtype)



########################################################################################################################
# Problem 4-2. Implement Dataloader
########################################################################################################################
def get_dataloader(root, transform, batch_size):
    data = dataset(root, train=True, transform=transform)

    # P4.4. Use `DataLoader` module for mini-batching train and test datasets.
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader
