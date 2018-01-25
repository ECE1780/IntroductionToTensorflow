import numpy as np
from PIL import Image
import os
import random


class MnistDataset:

    def __init__(self, data_dir):
        self.cursor = 0

        self.data = []
        for digit in range(10):
            images = os.listdir(os.path.join(data_dir, str(digit)))
            images = [os.path.join(data_dir, str(digit), image) for image in images]
            for image in images:
                self.data.append({'image': image, 'label': digit})
        random.shuffle(self.data)

    def sample(self, batch_size):
        batch = self.data[self.cursor:self.cursor+batch_size]
        if len(batch) < batch_size:     # Corner case f we need to wrap around the data
            batch += self.data[0:(self.cursor+batch_size)%len(self.data)]
        self.cursor = (self.cursor + batch_size) % len(self.data)

        images = [sample['image'] for sample in batch]
        images = [MnistDataset.read_image(image) for image in images]
        images = np.stack(images)

        labels = [sample['label'] for sample in batch]
        labels = [MnistDataset.one_hot(label) for label in labels]
        labels = np.stack(labels)

        return images, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)              # Image object
        image = np.array(image)                     # Multi-dimentional array
        image = image.astype(np.float32)            # uint to float
        image = image / 255                         # 0-255 -> 0-1
        image = image.reshape(image.shape + (1,))   # (width, height) -> (width, heigh, 1)
        return image

    @staticmethod
    def one_hot(label):
        encoded_label = np.zeros(10)
        encoded_label[label] = 1
        return encoded_label
