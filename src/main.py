"""Main module for the application."""

from conv.standard import Standard
from conv.threaded import Threaded
from loader.service import Loader

if __name__ == "__main__":
    image_path = "images/dog.jpg"
    image = Loader.get_valid_image(image_path)
    
    kernel = [
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ]

    standard_conv = Standard(kernel)
    convolved_image = standard_conv.run(image)

    threaded_conv = Threaded(kernel)
    convolved_image_threaded = threaded_conv.run(image, num_threads=16)
