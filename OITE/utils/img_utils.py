import cv2
import numpy as np

class imgUtils:
    """
    A utility class for image processing.

    Methods
    -------
    get_img_processed(img_path)
        Loads, processes, and returns the sharpened and binary versions of the input image.

    cutImage(image, rect)
        Cuts out a portion of the image defined by a rectangle.

    wipe_classes(img, classes)
        Removes the areas occupied by OntoUML classes from the image by drawing white rectangles over them.
    """

    @staticmethod
    def get_img_processed(img_path):
        """
        Loads and processes an image to create a sharpened and binary version.

        Parameters
        ----------
        img_path : str
            The path to the image file.

        Returns
        -------
        tuple
            A tuple containing the sharpened image and the binary image.
        """
        # Load the image
        image = cv2.imread(img_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for sharpening
        blurred = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=3, sigmaY=3)
        sharpened = cv2.addWeighted(gray_image, 1.5, blurred, -0.5, 0)
        
        # Perform threshold conversion (every pixel with value < 150 goes white, the rest black)
        _, binary_image = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)

        return sharpened, binary_image
    
    @staticmethod
    def cutImage(image, rect):
        """
        Cuts out a portion of the image defined by a rectangle.

        Parameters
        ----------
        image : numpy.ndarray
            The image from which to cut out the portion.
        rect : Rectangle
            The rectangle defining the portion to cut out.

        Returns
        -------
        numpy.ndarray
            The cut-out portion of the image.
        """
        return image[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]
    
    @staticmethod
    def wipe_classes(img, classes):
        """
        Removes the areas occupied by OntoUML classes from the image by drawing white rectangles over them.

        Parameters
        ----------
        img : numpy.ndarray
            The image from which to wipe the OntoUML classes.
        classes : list
            A list of OntoUMLClass objects whose areas are to be wiped from the image.

        Returns
        -------
        numpy.ndarray
            The image with the areas occupied by OntoUML classes wiped clean.
        """
        no_class_img = img.copy()
        for c in classes:
            cv2.rectangle(no_class_img, (c.whole.x - 5 , c.whole.y - 5), 
                          (c.whole.x + c.whole.width + 5, c.whole.y + c.whole.height + 5), 
                          (255, 255, 255), -1)
        return no_class_img