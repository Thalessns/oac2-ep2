"""Abstract base classes for convolution operations."""

import numpy as np
from abc import ABC, abstractmethod


class Conv2D(ABC):
    """Abstract base class for 2D convolution operations."""

    kernel: np.ndarray

    def __init__(self, kernel: list[list]) -> None:
        """Initialize Conv2D class.
        
        Args:
            kernel (list[list]): Kernel that will be used.
        """
        self.kernel = np.array(kernel)

    @abstractmethod
    def run(self, image) -> np.ndarray:
        """Run convolution operation on the given image.
        
        Args:
            image (Image): Image to apply convolution on.
        
        Returns:
            np.ndarray: Convolved image.
        """
        pass
