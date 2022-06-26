# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py

import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

PARAMETER_MAX = 10


# the fixmatch_augment_pool includes:
#     AutoContrast
#     Brightness
#     Color
#     Contrast
#     Cutout
#     CutoutAbs
#     Equalize
#     Identity
#     Invert
#     Posterize
#     Rotate
#     Sharpness
#     ShearX
#     Solarize
#     SolarizeAdd
#     TranslateX
#     TranslateY
# RandAugmentMC(n,m)
#     choose n augmentation for the pool
#     and repeat m times


def AutoContrast(img, **kwarg):
    return ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


class Cutout:
    def __init__(self, vmin=0.4, vmax=0.9, color=(127, 127, 127)):
        self.vmin = vmin
        self.vmax = vmax
        self.color = color
    
    def _pil_op(self, im):
        W, H = im.size
        w, h = round(W*random.uniform(self.vmin, self.vmax)), round(H*random.uniform(self.vmin, self.vmax))
        x, y = random.randint(0, W - w), random.randint(0, H - h)
        r = (w*h) / (W*H)
        if r>=0.5:
            background = Image.new("RGB", im.size, self.color)
            foreground = im.copy().crop((x, y, x+w, y+h))
            background.paste(foreground, (x, y))
        else:
            background = im.copy()
            # foreground = im.copy().crop((x, y, x+w, y+h)).resize((4, 4)).resize((w, h))
            # background.paste(foreground, (x, y))
            ImageDraw.Draw(background).rectangle((x, y, x+w, y+h), self.color)

        return background
    
    def __call__(self, im):
        return self._pil_op(im)


def Equalize(img, **kwarg):
    return ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()
        self.cutout = Cutout(0.4, 0.9)

    def __repr__(self):
        return f"RandAugmentMC(n=self.n, m=self.m)"

    def __str__(self):
        return self.__repr__()
    
    def __call__(self, img):
        l = max(img.size)
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(l*0.5))
        # img = self.cutout(img)
        return img


from numpy.fft import fft2, fftshift, ifftshift, ifft2

class FreqFliter:
    def __init__(self, scale=None):
        self.scale = scale

    def gen_mask(self, shape, scale, inverse=False):
        w, h = shape
        x, y = np.ogrid[:w, :h]
        cx, cy = w//2, h//2
        distence = (x - cx)**2 + (y - cy)**2
        radius = min(shape)*scale/2
        mask = (distence<=radius**2)
        if inverse:
            mask = ~mask
        return mask

    def image_filter(self, image, mask):
        H, W, C = image.shape
        mask = np.tile(mask[..., None], reps=(1, 1, C))
        t = fft2(image, axes=(0, 1))
        t = fftshift(t, axes=(0, 1))
        t = ifftshift(t*mask, axes=(0, 1))
        t = ifft2(t, axes=(0, 1))
        t = np.clip(np.abs(t), 0, 255).astype("uint8")
        return t

    def _pil_op(self, im):
        img = np.asarray(im)
        if isinstance(self.scale, float):
            scale = self.scale
        elif isinstance(self.scale, list):
            scale = random.choice(self.scale)
        else:
            scale = random.random()
        mask = self.gen_mask(im.size, scale)
        img = self.image_filter(img, mask)
        return Image.fromarray(img)
    
    def __call__(self, im):
        return self._pil_op(im)