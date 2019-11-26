
import torch
import numpy as np
class SignDataset(torch.utils.data.Dataset):
    # https://alexanderkurakin.blogspot.com/2019/01/pytorch-data-loading-preprocessing.html
    def __init__(self, imgs, labels,n_classes, transform = None):
        """
        Args:
            csv_file (string): Путь к csv файлу с аннотациями.
            root_dir (string): Каталог со всеми изображениями.
            transform (callable, optional): optional transform.
        """
        self.imgs = imgs.astype("float32") / 255.0
        # self.labels = labels.astype("float32") / (n_classes*1.0)
        self.labels = labels.astype("long")
        self.imgs = np.reshape(self.imgs,(len(self.imgs),3,32,32))
        self.imgs = torch.from_numpy(self.imgs)
        self.labels = torch.from_numpy(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # sample = {'image': self.imgs[idx], 'landmarks': self.labels[idx]}
        sample = self.imgs[idx], self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample
