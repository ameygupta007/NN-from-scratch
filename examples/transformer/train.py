import numpy as np
from minigrad import Tensor, Transformer, save, load
import minigrad.optim as optim
from data import load_data

import os
import psutil
import gc
import argparse
import time

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
        model : Transformer, data, vocab_size, block_size, batch_size, steps, save_path=None, save_every=0, log_every=50
):
        
    params = model.parameters()
    optimiser = optim.Adam(params)
    loss_vals = []
    model.train()
    

    print_header()
    start = time.time()
    try:
        for step in range(1, steps+1):

            x,y = get_batch(data, block_size, batch_size)
            y = one_hot(y, vocab_size)

            pred = model(x)

            loss = pred.softmax_cross_entropy(y)
            loss_vals.append(loss.data)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            gc.collect()

            if step == 1 or step % log_every == 0:
                recent_loss = loss_vals[-log_every:]
                av_loss = sum(recent_loss) / len(recent_loss)
                print_row(step, av_loss, get_memory_usage(), time.time() - start)

            if save_path and save_every and step % save_every == 0:
                save(model, save_path)
                
    except KeyboardInterrupt:
        print("\nInterrupted...")

    finally:
        if save_path:
            save(model, save_path)
            print(f"Saved --> {save_path}")

    return model

COLS = (
    ("steps", 8),
    ("loss", 10),
    ("mem/mb", 8),
    ("time/step", 10),
    ("elapsed",8),
)
SEP = " | "
def print_header():
    print(SEP.join(f"{name:>{width}}" for name, width in COLS))
    print("-+-".join("-"*width for _, width in COLS))

def print_row(steps, loss, mem, elapsed):
    vals = [f"{steps:>8}", f"{loss:>10.4f}", f"{mem:>8.0f}", f"{elapsed/steps:>10.2g}", f"{elapsed:>8.2f}",]
    print(SEP.join(vals))


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
    p = argparse.ArgumentParser( prog='TRAIN')
    p.add_argument('-r', '--resume', metavar='PATH', help='load weights from npz and continue')
    p.add_argument('--save', metavar='PATH', default='models/chkpt.npz')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--block-size', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--train-chars', type=int, default=0, help='truncate the corpus (0 uses all of it)')
    p.add_argument('--log-every', type=int, default=50)
    args = p.parse_args()
    
    train_data, test_data, encode, decode, vocab_size  = load_data()

    if args.resume:
        model = load(args.resume, Transformer)
        cfg = model.config()
        if cfg['vocab_size'] != vocab_size:
            raise SystemExit(f"vocab mismatch: checkpoint {cfg['vocab_size']}, data {vocab_size}")
        print(f"Resumed from {args.resume}: {cfg}")
    else:
        model = Transformer(vocab_size, 128, 4, 4, dropout_p=0.2)

    data = train_data if args.train_chars == 0 else train_data[:args.train_chars]

    train(model, data, vocab_size, 
          args.block_size, args.batch_size, args.steps, 
          save_path=args.save, save_every=args.save_every, log_every=args.log_every)
    

