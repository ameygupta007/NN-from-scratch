import gc
import time

import numpy as np
from minigrad import SGD, step_decay

def predict(model, x):
    was_training = model.training
    model.eval() 
    try:
        Z = model(x)
        return np.argmax(Z.data, axis=1)
    finally:
        if was_training:
            model.train()

def evaluate(model, x, y):
    # give accuracy of model on data x with expected output y
    # y: np array of outputs we want
    preds = predict(model, x)
    return float(np.sum(np.equal(preds, y)) / len(y))

def train(
    model, x_pool, y_pool,
    x_test=None, y_test=None,
    num_epochs=5000, batch_size=128, steps_per_epoch=None,
    lr=1.0, lr_decay_at=(0.5, 0.75), lr_decay_factor=0.1,
    log_every=100, gc_every=20,
):
    # training loop: train model on x_pool, y_pool
    N_pool = len(x_pool)
    if steps_per_epoch is None:
        steps_per_epoch = 60000 // batch_size

    params = model.parameters()
    loss_vals = []

    opt = SGD(params, lr=1.0)
    sched = step_decay(num_epochs, lr, lr_decay_at, lr_decay_factor)

    model.train()

    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        _lr = opt.lr
        opt.lr = sched(epoch)
        if opt.lr != _lr:
            print('STEP CHANGE: new learning rate is', opt.lr)
        
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, N_pool, size=batch_size) # chance of duplicates is negligible and not that important
            xb, yb = x_pool[idx], y_pool[idx]

            # forwards
            Z = model(xb)
            loss = Z.softmax_cross_entropy(yb)
            loss_vals.append(loss.data)

            # remember to zero grad!
            opt.zero_grad()
            # backwards, update weights
            loss.backward()
            opt.step()

        # logging
        if epoch == 0:
            print(epoch, sum(loss_vals[-50:]) / 50)
        if (epoch % log_every == 0 and epoch > 0) or epoch==num_epochs:
            avg_loss = sum(loss_vals[-50:]) / 50
            per100 = (time.time() - start_time) * 100 / epoch
            mean_grad = np.abs(model.layers[0].W.grad).mean()
            msg = f'{epoch} {avg_loss:.4f} {per100:.1f}s/100epochs Mean grad: {mean_grad:.6f}'
            if x_test is not None and y_test is not None:
                acc = evaluate(model, x_test, y_test)
                model.train()
                msg += f' Test accuracy: {acc}'
            print(msg)

        # run gc to prevent memory usage exploding
        if epoch % gc_every == 0:
            gc.collect()

    return loss_vals
