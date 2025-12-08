"""Module for convolution operations."""

import numpy as np
import time
import threading

from PIL.Image import Image
from conv.abstract import Conv2D


class Threaded(Conv2D):
    """Class for 2D convolution operations."""

    kernel: np.ndarray

    def run(self, image: Image, num_threads: int) -> np.ndarray:
        """Run convolution operation on the given image.
        
        Args:
            image (Image): Image to apply convolution on.
            num_threads (int): Number of threads to use.

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

        def process_lines(start_row, end_row):
            nonlocal output

            for i in range(img_h):
                for j in range(img_w):
                    pixel = padded_img[i:i+kerner_h, j:j+kerner_w]
                    output[i, j] = np.sum(pixel * self.kernel)

        rows_per_thread = img_h // num_threads
        threads = []

        start_time = time.time()

        for t in range(num_threads):
            start = t * rows_per_thread
            end = (t + 1) * rows_per_thread if t != num_threads - 1 else img_h
            thread = threading.Thread(target=process_lines, args=(start, end))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        end_time = time.time()

        print(f"Threaded Convolution took {end_time - start_time:.6f} seconds.")

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
