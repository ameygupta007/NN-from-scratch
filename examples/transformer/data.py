from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / 'data'

def load_data(dir=DATA_DIR):
    '''
    char level tokenization. Load tinyshakespeare dataset.

    Returns: train_data, test_data, encode, deocde, vocab_size.
    '''
    text = (Path(dir) / 'tinyshakespeare.txt').read_text(encoding='utf-8')
    chars = sorted(set(text))

    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}

    encode = lambda s:  [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = encode(text)
    n = int(len(text) * 0.9)
    return data[:n], data[n:], encode, decode, len(chars)


if __name__ == '__main__':
    load_data()