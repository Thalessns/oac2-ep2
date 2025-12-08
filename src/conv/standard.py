"""Module for convolution operations."""

import numpy as np
import time

from PIL.Image import Image
from conv.abstract import Conv2D


class Standard(Conv2D):
    """Class for 2D convolution operations."""

    kernel: np.ndarray

    def run(self, image: Image) -> np.ndarray:
        """Run convolution operation on the given image.
        
        Args:
            image (Image): Image to apply convolution on.
        
        Returns:
            np.ndarray: Convolved image.
        """
        np_image = np.array(image)
        img_w, img_h = np_image.shape
        kerner_w, kerner_h = self.kernel.shape
        padding_w = kerner_w // 2
        padding_h = kerner_h // 2

        padded_img = np.pad(
            np_image, 
            ((padding_h, padding_h), 
             (padding_w, padding_w)), 
             mode='constant'
        )
        output = np.zeros_like(np_image)

        start_time = time.time()

        for i in range(img_h):
            for j in range(img_w):
                pixel = padded_img[i:i+kerner_h, j:j+kerner_w]
                output[i, j] = np.sum(pixel * self.kernel)

        end_time = time.time()
        print(f"Standard Convolution took {end_time - start_time:.6f} seconds.")

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
