import os
import scipy.misc
import numpy as np
from glob import glob
import imageio


class PicPreProcessing:

    def __init__(self):
        # self.data_name = path
        self.source_shape = (96, 96, 3)
        self.resize_shape = (64, 64, 3)
        # self.resize_shape = (64, 64, 28)
        self.crop = True
        # self.img_shape = self.source_shape if not self.crop else self.resize_shape
        self.img_shape = self.resize_shape
        # self.img_shape = (48, 48, 3)
        # self.img_list = self._get_img_list()
        # self.batch_size = 125
        # self.batch_shape = (self.batch_size,) + self.img_shape
        # self.chunk_size = len(self.img_list) // self.batch_size

    def _get_img_list(self,data_name):
        # path = os.path.join(os.getcwd(), self.data_name, '*.jpg')
        path = os.path.join(os.getcwd(), data_name, '*.jpg')
        return glob(path)

    def _get_img(self, name):
        # assert name in self.img_list
        img = scipy.misc.imread(name).astype(np.float32)
        # print("read img:" + name)
        # assert img.shape == self.source_shape
        return self._resize(img) if self.crop else img

    def _resize(self, img):
        h, w = img.shape[:2]
        resize_h, resize_w = self.resize_shape[:2]
        crop_h, crop_w = self.source_shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        cropped_image = scipy.misc.imresize(img[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])
        return np.array(cropped_image) / 127.5 - 1.

    @staticmethod
    def save_img(image, path):
        scipy.misc.imsave(path, image)
        return True

    def batches(self):
        start = 0
        end = self.batch_size
        for _ in range(self.chunk_size):
            name_list = self.img_list[start:end]
            imgs = [self._get_img(name) for name in name_list]
            batches = np.zeros(self.batch_shape)
            batches[::] = imgs
            yield batches
            start += self.batch_size
            end += self.batch_size
        print(name_list)
        print(imgs)


if __name__ == '__main__':
    avatar = Avatar()
    print(avatar.img_shape)
    name_list = avatar.img_list
    batch = avatar.batches()
    imgs = [avatar._get_img(name) for name in name_list]
    print(imgs)

    b = next(batch)
    # for num in range(len(b)):
    #     avatar.save_img(b[num], 'samples'+os.sep+str(num)+'.jpg')
