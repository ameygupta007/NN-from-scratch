import numpy as np
from data import load_data
from minigrad import load, Transformer
import argparse

def generate(model, length, encode, decode, block_size, start="r"):
    model.eval()
    out = list(encode(start))

    for i in range(length):
        context = np.asarray(out[-block_size:])[None,...]
        logits = model(context).softmax(axis=-1).data[0,-1]
        pred = np.random.choice(np.arange(len(logits)), p=logits)
        out.append(pred)

    return decode(out)

def load_transformer(path):
    return load(path, Transformer)

if __name__ == '__main__':

    p = argparse.ArgumentParser(prog="SAMPLE")
    p.add_argument('model_path')
    p.add_argument('-l', '--length', type=int, default=200)
    p.add_argument('-s', '--start', default='\n')

    args = p.parse_args()

    *_, encode, decode, vocab_size = load_data()

    t = load_transformer(args.model_path)
    predicted = generate(t, args.length, encode, decode, 64, start=args.start)
    print("======PROMPT=======")
    print(args.start)
    print("=====GENERATED=====")
    print(predicted[len(args.start):])