import struct
import numpy as np

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f'Invalid magic number {magic} in {filename}'
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
        return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f'Invalid magic number {magic} in {filename}'
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Example usage:
train_images = load_images('train-images-idx3-ubyte')
train_labels = load_labels('train-labels-idx1-ubyte')
test_images  = load_images('t10k-images-idx3-ubyte')
test_labels  = load_labels('t10k-labels-idx1-ubyte')

# Print some stats
print(f'Train images shape: {train_images.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Test images shape: {test_images.shape}')
print(f'Test labels shape: {test_labels.shape}')

def save_mnist_as_text(images, labels, filename):
    num_samples = images.shape[0]
    with open(filename, 'w') as f:
        for i in range(num_samples):
            label = labels[i]
            pixels = images[i].flatten()
            pixels_str = ' '.join(str(p) for p in pixels)
            line = f"{label} {pixels_str}\n"
            f.write(line)

# Usage example (using your already loaded data):
save_mnist_as_text(train_images, train_labels, 'train_data.txt')
save_mnist_as_text(test_images, test_labels, 'test_data.txt')

