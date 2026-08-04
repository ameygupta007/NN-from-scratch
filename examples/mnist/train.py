import gc
import time

import numpy as np

def evaluate(model, x, y):
    # give accuracy of model on data x with expected output y
    # y: np array of outputs we want
    was_training = model.training
    model.training = False
    try:
        Z = model(x)
        p = np.argmax(Z.data, axis=1)
        return float(np.sum(np.equal(p, y)) / len(y))
    finally:
        model.training = was_training

def train(
    model, x_pool, y_pool,
    x_test=None, y_test=None,
    num_epochs=5001, batch_size=128, steps_per_epoch=None,
    lr=1.0, lr_decay_at=(0.5, 0.75), lr_decay_factor=0.1,
    log_every=100, gc_every=20,
):
    # training loop: train model on x_pool, y_pool
    N_pool = len(x_pool)
    if steps_per_epoch is None:
        steps_per_epoch = 60000 // batch_size

    params = model.parameters()
    loss_vals = []
    h = lr
    decay_epochs = {int(num_epochs * frac) for frac in lr_decay_at}
    start_time = time.time()

    model.training = True
    for epoch in range(num_epochs):
        if epoch in decay_epochs and epoch > 0:
            h *= lr_decay_factor
            print('STEP CHANGE: new learning rate is', h)

        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, N_pool, size=batch_size) # chance of duplicates is negligible and not that important
            xb, yb = x_pool[idx], y_pool[idx]

            # forwards
            Z = model(xb)
            loss = Z.softmax_cross_entropy(yb)
            loss_vals.append(loss.data)

            # remember to zero grad!
            for p in params:
                p.grad = np.zeros_like(p.grad)

            # backwards, update weights
            loss.backward()
            for p in params:
                p.data -= p.grad * h

        if epoch == 0:
            print(epoch, sum(loss_vals[-50:]) / 50)
        if epoch % log_every == 0 and epoch > 0:
            avg_loss = sum(loss_vals[-50:]) / 50
            per100 = (time.time() - start_time) * 100 / epoch
            mean_grad = np.abs(model.layers[0].W.grad).mean()
            msg = f'{epoch} {avg_loss:.4f} {per100:.1f}s/100epochs Mean grad: {mean_grad:.6f}'
            if x_test is not None and y_test is not None:
                acc = evaluate(model, x_test, y_test)
                model.training = True
                msg += f' Test accuracy: {acc}'
            print(msg)

        if epoch % gc_every == 0:
            gc.collect()

    return loss_vals
