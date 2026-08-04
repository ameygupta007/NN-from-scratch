import numpy as np

def _bilinear_sample(img, src_y, src_x):
    # img: 2D numpy array, e.g. 28x28
    # src_y, src_x: 2D numpy arrays - for every output pixel, fractional source coordinates 
    # return a new image by sampling src_x, src_y from img
    h, w = img.shape
    # 4 coords surrounding src_x, src_y
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int) 
    y1 = y0 + 1

    # fractional part
    ty = src_y - y0
    tx = src_x - x0
    # clip so that coords are in img range
    valid = (y0 >= 0) & (y1 < h) & (x0 >= 0) & (x1 < w)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)

    # the 4 samples for each output pixel
    a = img[y0c, x0c]
    b = img[y0c, x1c]
    c = img[y1c, x0c]
    d = img[y1c, x1c]

    # bilinear interpolation: x
    top = a * (1 - tx) + b * tx
    bot = c * (1 - tx) + d * tx
    # if out of range fill in black (background)
    return np.where(valid, top * (1 - ty) + bot * ty, 0.0)

def random_shift(img, max_pixels=2):
    dy = np.random.uniform(-max_pixels, max_pixels)
    dx = np.random.uniform(-max_pixels, max_pixels)
    h, w = img.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return _bilinear_sample(img, yy - dy, xx - dx)

def random_rotate(img, max_degrees=10):
    theta = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    h, w = img.shape
    # centre
    cy, cx = (h - 1) / 2, (w - 1) / 2
    # shift image so centre is at 0,0
    yy, xx = np.meshgrid(np.arange(h) - cy, np.arange(w) - cx, indexing='ij')
    # inverse rotation transformation: y_sample = y*cos(-t) + x*sin(-t), x_sample = x*cos(-t) - y*sin(-t)
    src_y = cos_t * yy - sin_t * xx + cy
    src_x = sin_t * yy + cos_t * xx + cx

    return _bilinear_sample(img, src_y, src_x)

def random_scale(img, min_factor=0.9, max_factor=1.1):
    factor = np.random.uniform(min_factor, max_factor)
    h, w = img.shape
    # centre
    cy, cx = (h - 1) / 2, (w - 1) / 2
    # shift so centre is at origin
    yy, xx = np.meshgrid(np.arange(h) - cy, np.arange(w) - cx, indexing='ij')
    # inverse scale transform
    src_y = yy / factor + cy
    src_x = xx / factor + cx
    return _bilinear_sample(img, src_y, src_x)

def augment_image(img, shift_px=2, rotate_deg=10, scale_range=(0.9, 1.1)):
    img = random_shift(img, max_pixels=shift_px)
    img = random_rotate(img, max_degrees=rotate_deg)
    img = random_scale(img, min_factor=scale_range[0], max_factor=scale_range[1])
    return np.clip(img, 0.0, 1.0)

def augment_batch(xb, img_shape=(28, 28), shift_px=2, rotate_deg=10, scale_range=(0.9, 1.1)):
    # xb: np array of flat images
    batch_size = xb.shape[0]
    imgs = xb.reshape(batch_size, *img_shape)
    # could vectorize this
    out = np.empty_like(imgs)
    for i in range(batch_size):
        out[i] = augment_image(imgs[i], shift_px, rotate_deg, scale_range)
    # flatten images again
    return out.reshape(batch_size, -1)
