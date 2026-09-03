import numpy as np
from minigrad import Tensor, Transformer, save, load
import minigrad.optim as optim
from data import load_data

import os
import psutil
import gc
import argparse

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

def generate(model : Transformer, length, encode, decode, block_size, start="r"):
    model.eval()
    out = list(encode(start))

    for i in range(length):
        context = np.asarray(out[-block_size:])[None,...]
        logits = model(context).softmax(axis=-1).data[0,-1]
        pred = np.random.choice(np.arange(len(logits)), p=logits)
        out.append(pred)

    return decode(out)


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
        loss_vals.append(loss.data)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


        if step % 1 == 0:
            gc.collect()
        if step > 0 and step % 50 == 0:
            print(step, sum(loss_vals[-50:]) / 50, get_memory_usage(), f"live Tensors: {tensor_census()} MB")
        if step == 0:
            print(0, loss.data)
    return model

def get_memory_usage():
    # Get the process ID of the current Python script
    process = psutil.Process(os.getpid())
    
    # Get Resident Set Size (RSS) memory in bytes and convert to Megabytes
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / (1024 * 1024)
    return mem_mb

def tensor_census():
    ts = [o for o in gc.get_objects() if isinstance(o, Tensor)]
    mb = sum(t.data.nbytes + t.grad.nbytes for t in ts) / 2**20
    return len(ts), mb

if __name__ == '__main__':
    train_data, test_data, encode, decode, vocab_size  = load_data()
    t = Transformer(vocab_size, 128, 4, 4, dropout_p=0.2)
    path = 'test_train.npz'
    t = load(path, Transformer)
    try:
        train(t, train_data, vocab_size, block_size=64, batch_size=10, steps=2000)
    except KeyboardInterrupt:
        print(f"\nInterrupted... saving to {path}")
    finally:
        save(t, path)

    print('---------')
    print('SAMPLE:')
    print('---------')
    t = load(path, Transformer)
    seed = '''\n'''
    predicted = generate(t, 500, encode, decode, 64, start=seed)
    print(seed)
    print("-"*10)
    print(predicted[len(seed):])
