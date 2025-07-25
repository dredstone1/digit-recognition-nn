import urllib.request
import gzip
import os
import struct
import numpy as np

BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images':  't10k-images-idx3-ubyte.gz',
    'test_labels':  't10k-labels-idx1-ubyte.gz'
}

def download_and_extract():
    for key, filename in FILES.items():
        print(f'Downloading {filename}...')
        url = BASE_URL + filename
        urllib.request.urlretrieve(url, filename)
        extracted = filename[:-3]
        print(f'Extracting {filename} -> {extracted}')
        with gzip.open(filename, 'rb') as f_in, open(extracted, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(filename)  # cleanup .gz file

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f'Invalid magic number {magic}'
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((num_images, rows, cols))

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f'Invalid magic number {magic}'
        return np.frombuffer(f.read(), dtype=np.uint8)

def save_mnist_as_text(images, labels, filename):
    with open(filename, 'w') as f:
        f.write(f"{len(images)} 784\n")

        for img, lbl in zip(images, labels):
            flat = img.flatten()
            f.write(f"{lbl} {' '.join(map(str, flat))}\n")

def main():
    download_and_extract()

    train_images = load_images('train-images-idx3-ubyte')
    train_labels = load_labels('train-labels-idx1-ubyte')
    test_images  = load_images('t10k-images-idx3-ubyte')
    test_labels  = load_labels('t10k-labels-idx1-ubyte')

    print(f'Train images: {train_images.shape}')
    print(f'Train labels: {train_labels.shape}')
    print(f'Test images:  {test_images.shape}')
    print(f'Test labels:  {test_labels.shape}')

    save_mnist_as_text(train_images, train_labels, 'train_data.nndb')
    save_mnist_as_text(test_images, test_labels, 'test_data.nndb')

    for fname in [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]:
        if os.path.exists(fname):
            os.remove(fname)
            print(f'Removed {fname}')

if __name__ == "__main__":
    main()

