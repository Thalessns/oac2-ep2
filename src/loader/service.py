"""Module for load images."""

from PIL import Image


class Loader:
    """Class for loading images."""
    valid_resolutions = [(512, 512), (1024, 1024), (4096, 4096)]

    @classmethod
    def get_valid_image(cls, image_path: str) -> Image.Image:
        """Load an image from the given path.
        
        Args:
            image_path (str): Path to the image file.
        """
        image = cls.load_gray_scaled_image(image_path)
        return cls.convert_to_valid_resolution(image)

    @classmethod
    def load_gray_scaled_image(cls, image_path: str) -> Image.Image:
        """Load a grayscale image from the given path.
        
        Args:
            image_path (str): Path to the image file.
        """
        return Image.open(image_path).convert("L")

    @classmethod
    def convert_to_valid_resolution(cls, image: Image.Image) -> Image.Image:
        """Convert the image to a valid resolution if necessary.
        
        Args:
            image (Image.Image): The image to be resized.
            valid_resolution (tuple): The target resolution as (width, height).
        """
        if image.size in cls.valid_resolutions:
            return image
        return image.resize(cls.valid_resolutions[1])
