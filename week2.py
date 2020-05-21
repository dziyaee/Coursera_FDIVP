# Week 2 Homework, Questions 7 and 8

import os
from PIL import Image
import numpy as np
from numpy import log10
from dzlib.common.utils import info, stats
from scipy.ndimage.filters import convolve
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")


# Load image as numpy array
image_path = '/'.join([os.getcwd(), 'image1.gif'])
image1 = Image.open(image_path)
image1 = np.asarray(image1).astype(np.float32)
info(image1, 'input')
stats(image1, 'input')

# Convolve with 3x3 Average Filter
kernel = np.ones((3, 3))
kernel *= 1 / kernel.size
output1 = convolve(input=image1, weights=kernel, mode='nearest')

# MSE & PSNR
mse = (1 / image1.size) * np.sum((image1 - output1) ** 2)
max_i = 255
psnr1 = 10 * log10(max_i ** 2 / mse)
print(f"\n3x3 Average Filter PSNR: {psnr1:.3f}")

# Convolve with 5x5 Average Filter
kernel = np.ones((5, 5))
kernel *= 1 / kernel.size
output2 = convolve(input=image1, weights=kernel, mode='nearest')

# MSE & PSNR
mse = (1 / image1.size) * np.sum((image1 - output2) ** 2)
max_i = 255
psnr2 = 10 * log10(max_i ** 2 / mse)
print(f"5x5 Average Filter PSNR: {psnr2:.3f}")

# Plot
title = iter(['Input', f'3x3 Average Filter, PSNR: {psnr1:.3f}', f'5x5 Average Filter, PSNR: {psnr2:.3f}'])
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
for axis, image in zip(axes, [image1, output1, output2]):
    axis.imshow(image)
    axis.set_title(next(title))
plt.show(block=True)
