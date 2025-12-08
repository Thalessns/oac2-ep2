"""Module for convolution operations."""

import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
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

        def process_block(start_row: int, end_row: int) -> tuple:
            """Process image block.
            
            Args:
                start_row (int): The first row of the block.
                end_row (int): Last row of the block.

            Returns:
                tuple: returns a tuple with the start row and the block sums results.
            """
            block_result = np.zeros((end_row - start_row, img_w))
            for i_local, i_global in enumerate(range(start_row, end_row)):
                for j in range(img_w):
                    pixel = padded_img[i_global:i_global+kerner_h, j:j+kerner_w]
                    block_result[i_local, j] = np.sum(pixel * self.kernel)
            return start_row, block_result

        rows_per_thread = max(1, img_h // num_threads)
        threads = []

        for t in range(num_threads):
            start = t * rows_per_thread
            end = min((t + 1) * rows_per_thread, img_h) if t != num_threads - 1 else img_h
            if start < end:
                threads.append((start, end))

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_block, start, end) for start, end in threads]

            for future in as_completed(futures):
                start_row, result = future.result()
                output[start_row:start_row + result.shape[0], :] = result

        end_time = time.time()

        print(f"Threaded Convolution took {end_time - start_time:.6f} seconds.")

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
