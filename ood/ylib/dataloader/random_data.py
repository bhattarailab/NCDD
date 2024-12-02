from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.vision import VisionDataset
from scipy.fftpack import idct

class GaussianRandom(VisionDataset):
    def __init__(
            self, image_size, data_size
    ) :
        self.image_size = image_size
        self.data_size = data_size
        super(GaussianRandom, self).__init__('')

    def __getitem__(self, index):
        return np.random.randn(3, self.image_size, self.image_size).astype(float), np.random.randint(0,10)

    def __len__(self):
        return self.data_size


class LowFreqRandom(VisionDataset):
    def __init__(
            self, image_size, data_size
    ) :
        self.image_size = image_size
        self.data_size = data_size
        super(LowFreqRandom, self).__init__('')

    def __getitem__(self, index):
        x = np.random.randn(3, self.image_size, self.image_size)
        z = self.idct2d(x, ratio=0.5)
        # plt.imshow(np.rollaxis(z[0].numpy(), 0, 3))
        return z, np.random.randint(0,10)

    def idct2d(self, x, ratio=0.5):
        z = np.zeros_like(x)
        mask = np.zeros_like(x)
        mask[:, :int(x.shape[1] * ratio), :int(x.shape[2] * ratio)] = 1
        x *= mask
        z = idct(idct(x, axis=2, norm='ortho'), axis=1, norm='ortho')
        return z

    def __len__(self):
        return self.data_size

