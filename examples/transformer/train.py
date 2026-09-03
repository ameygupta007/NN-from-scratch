from minigrad import Module, Transformer
import numpy as np
from data import load_data
import minigrad.optim as optim
import gc

import os
import psutil

def get_batch(data, block_size, batch_size):
    data = np.asarray(data, dtype=np.int64)

    starts = np.random.randint(0, len(data)-block_size, size=batch_size)
    offsets = np.arange(block_size)
    idx = starts[:, None] + offsets

    x = data[idx]
    y = data[idx+1]
    return x,y

def one_hot(y_ints, num_classes):
    y_ints = np.asarray(y_ints, dtype=np.int16)
    y_oh = np.zeros((*y_ints.shape, num_classes), dtype=np.bool)

    np.put_along_axis(y_oh, y_ints[..., None], 1, axis=-1)
    return y_oh

def train(
        model : Transformer, data, vocab_size, block_size, batch_size, steps
):
        
    params = model.parameters()
    optimiser = optim.Adam(params)
    loss_vals = []
    model.train()
    for step in range(steps):

        x,y = get_batch(data, block_size, batch_size)
        y = one_hot(y, vocab_size)

        pred = model(x)

        loss = pred.softmax_cross_entropy(y)
        loss_vals.append(loss)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


        if step % 10 == 0:
            print(step, loss.data, get_memory_usage())
        if step % 20 == 0:
            gc.collect()

    return model

def get_memory_usage():
    # Get the process ID of the current Python script
    process = psutil.Process(os.getpid())
    
    # Get Resident Set Size (RSS) memory in bytes and convert to Megabytes
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / (1024 * 1024)
    return mem_mb


if __name__ == '__main__':
    train_data, test_data, encode, decode, vocab_size  = load_data()
    t = Transformer(vocab_size, 128, 4, 4, dropout_p=0.2)

    train(t, train_data[:1000], vocab_size, 64, 10, 10000)