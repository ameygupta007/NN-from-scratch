import os
import struct
import time
from os.path import join

import numpy as np

from augment import augment_batch

def _read_idx_images(path):
    with open(path, 'rb') as f:
        magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Magic number mismatch in {path}, expected 2051, got {magic}')
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(size, rows, cols)

def _read_idx_labels(path):
    with open(path, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Magic number mismatch in {path}, expected 2049, got {magic}')
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)

def load_mnist(data_dir='data'):
    # load test and train x,y
    x_train = _read_idx_images(join(data_dir, 'train-images.idx3-ubyte'))
    y_train = _read_idx_labels(join(data_dir, 'train-labels.idx1-ubyte'))
    x_test = _read_idx_images(join(data_dir, 't10k-images.idx3-ubyte'))
    y_test = _read_idx_labels(join(data_dir, 't10k-labels.idx1-ubyte'))

    # flatten images
    x_train = x_train.astype(np.float64).reshape(-1, 784) / 255.0
    x_test = x_test.astype(np.float64).reshape(-1, 784) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return x_train, y_train, x_test, y_test


def one_hot(y, num_classes=10):
    # one hot encoding for y training values 
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


def build_or_load_augmented_pool(
    x, y_encoded, path, K=4,
    shift_px=3, rotate_deg=20, scale_range=(0.8, 1.2),
):
    # create augmented copies of dataset for better generalisation - if already created, load them.
    # K: number of augmented copies (final pool = (K+1) x original)
    if os.path.exists(path):
        print(f'loading cached pool from {path}')
        data = np.load(path)
        x_pool = data['x'].astype(np.float64)
        y_pool = data['y']
    else:
        print(f'generating {K} augmented copies of x...')
        parts_x = [x.astype(np.float32)]
        parts_y = [y_encoded.astype(np.float32)]
        for k in range(K):
            t0 = time.time()
            parts_x.append(augment_batch(
                x, shift_px=shift_px, rotate_deg=rotate_deg, scale_range=scale_range,
            ).astype(np.float32))
            parts_y.append(y_encoded.astype(np.float32))
            print(f'copy {k+1}/{K} done in {time.time() - t0:.1f}s')
        x_pool = np.concatenate(parts_x)
        y_pool = np.concatenate(parts_y)
        np.savez_compressed(path, x=x_pool, y=y_pool)
        print(f'saved {path} ({os.path.getsize(path) / 1e6:.0f} MB on disk)')

    print(f'pool size: {len(x_pool):,} samples ({len(x_pool) / len(x):.0f}x original)')
    return x_pool, y_pool
