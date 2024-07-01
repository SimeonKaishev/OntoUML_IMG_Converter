import cv2
import numpy as np

class Rectangle:
    """
    A class used to represent a Rectangle.

    Attributes
    ----------
    image : any, optional
        An image associated with the rectangle (default is None)
    contour : any, optional
        The contour of the rectangle (default is None)
    approx_curve : any, optional
        The approximated curve of the rectangle (default is None)
    x : int
        The x-coordinate of the rectangle's top-left corner (default is 0)
    y : int
        The y-coordinate of the rectangle's top-left corner (default is 0)
    width : int
        The width of the rectangle (default is 0)
    height : int
        The height of the rectangle (default is 0)
    within_cls_obj : bool
        A flag indicating whether the rectangle is within a class object (default is False)
    """

    def __init__(self, image=None, contour=None, approx_curve=None, x=0, y=0, width=0, height=0):
        """
        Constructs all the necessary attributes for the rectangle object.

        Parameters
        ----------
        image : any, optional
            An image associated with the rectangle (default is None)
        contour : any, optional
            The contour of the rectangle (default is None)
        approx_curve : any, optional
            The approximated curve of the rectangle (default is None)
        x : int, optional
            The x-coordinate of the rectangle's top-left corner (default is 0)
        y : int, optional
            The y-coordinate of the rectangle's top-left corner (default is 0)
        width : int, optional
            The width of the rectangle (default is 0)
        height : int, optional
            The height of the rectangle (default is 0)
        """
        self.image = image
        self.contour = contour
        self.approx_curve = approx_curve
        self.within_cls_obj = False
        
        # Calculate the bounding box from the approx_curve
        if approx_curve is not None:
            x, y, w, h = cv2.boundingRect(approx_curve)
            self.x = x
            self.y = y
            self.width = w
            self.height = h
        else:
            self.x = x
            self.y = y
            self.width = width
            self.height = height

    def tl(self):
        """
        Returns the top-left corner of the rectangle.

        Returns
        -------
        tuple
            A tuple containing the x and y coordinates of the top-left corner.
        """
        return (self.x, self.y)
    
    def br(self):
        """
        Returns the bottom-right corner of the rectangle.

        Returns
        -------
        tuple
            A tuple containing the x and y coordinates of the bottom-right corner.
        """
        return (self.x + self.width, self.y + self.height)

    def clone(self):
        """
        Creates a clone of the rectangle.

        Returns
        -------
        Rectangle
            A new instance of the Rectangle class with the same attributes.
        """
        return Rectangle(self.image, self.contour, self.approx_curve)
    
    def print_rectangle(self):
        """
        Prints the attributes of the rectangle.
        """
        print(f"approx_curve: {self.approx_curve}")
        print(f"x: {self.x}")
        print(f"y: {self.y}")
        print(f"width: {self.width}")
        print(f"height: {self.height}")
        print(f"within_class: {self.within_cls_obj}")

    def contains(self, point):
        """
        Checks if a given point is within the rectangle.

        Parameters
        ----------
        point : tuple
            A tuple containing the x and y coordinates of the point.

        Returns
        -------
        bool
            True if the point is within the rectangle, False otherwise.
        """
        return (self.x <= point[0] <= self.x + self.width) and (self.y <= point[1] <= self.y + self.height)
    

    @staticmethod
    def make_points_arr(approx_curve):
        """
        Converts the approximated rectangle curve to an array of points.

        Parameters
        ----------
        approx_curve : any
            The approximated curve of the rectangle.

        Returns
        -------
        numpy.ndarray
            An array of points derived from the approximated curve.
        """
        points = []
        for p in approx_curve:
            points.append([p[0][0], p[0][1]])
        return np.array(points)

    @staticmethod
    def is_approx_rectangle(approx_curve, tol=35):
        """
        Check if the given set of points forms an approximate rectangle.

        Parameters
        ----------
        approx_curve : any
            The approximated curve of the rectangle.
        tol : float, optional
            Tolerance for angle approximation in degrees (default is 35).

        Returns
        -------
        bool
            True if the points form an approximate rectangle, False otherwise.
        """
        points = Rectangle.make_points_arr(approx_curve)
        # Ensure points are sorted by their x and y coordinates to maintain correct order
        points = points[np.lexsort((points[:, 1], points[:, 0]))]

        # Function to calculate Euclidean distance
        def distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        # Calculate distances between consecutive points
        d1 = distance(points[0], points[1])
        d2 = distance(points[1], points[2])
        d3 = distance(points[2], points[3])
        d4 = distance(points[3], points[0])

        # Calculate diagonal distances
        diag1 = distance(points[0], points[2])
        diag2 = distance(points[1], points[3])

        # Check if opposite sides are approximately equal
        sides_equal = np.isclose(d1, d3, 10.0) and np.isclose(d2, d4, 10.0)

        # Check if diagonals are approximately equal
        diagonals_equal = np.isclose(diag1, diag2, 15.0)

        # Function to calculate angle between three points (p1 -> p2 -> p3)
        def angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

        # Calculate angles at each corner
        angle1 = angle(points[0], points[1], points[2])
        angle2 = angle(points[1], points[2], points[3])
        angle3 = angle(points[2], points[3], points[0])
        angle4 = angle(points[3], points[0], points[1])

        # Check if all angles are approximately 90 degrees
        angles_90 = all(np.isclose(angle, 90, atol=tol) for angle in [angle1, angle2, angle3, angle4])
        
        # Final check for rectangle
        return sides_equal and diagonals_equal and angles_90
    
    @staticmethod
    def detect_rectangles(original_img, image_path):
        """
        Detects rectangles in an image, wipes them and returns the processed image and list of rectangles.

        Parameters
        ----------
        original_img : numpy.ndarray
            The original image in which to detect rectangles.
        image_path : str
            The path to the image file.

        Returns
        -------
        tuple
            A tuple containing the processed image and a list of Rectangle objects detected in the image.
        """
        img = original_img.copy()
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        all_rect_areas = []
        rect_contours = []

        # Calculate area thresholds
        cd_area = img.shape[0] * img.shape[1]
        min_cls_area = 60
        max_cls_area = cd_area * 0.5
        
        for contour in contours:
            # Calculate contour area
            contour_area = cv2.contourArea(contour)
            if contour_area < min_cls_area or contour_area > max_cls_area:
                continue
            # Approximate the contour to a polygon
            curve = np.array(contour, dtype=np.float32)
            approx_curve = cv2.approxPolyDP(curve, 0.01 * cv2.arcLength(curve, True), True)

            # If the approximation has 4 vertices, it is considered a rectangle
            if len(approx_curve) == 4 and Rectangle.is_approx_rectangle(approx_curve):
                # Save the rectangle
                rect = Rectangle(img.copy(), contour, approx_curve)
                all_rect_areas.append(rect)
                rect_contours.append(contour)
                
                # Fill the rectangle to remove it from the image
                cv2.fillConvexPoly(img, contour, (255, 255, 255))

        # TODO remove after testing
        # Display all the recognized rectangles
        image_all = cv2.imread(image_path)
        for contour in rect_contours:
            image = cv2.imread(image_path)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 3)
            cv2.drawContours(image_all, [contour], -1, (0, 0, 255), 3)
            # cv2.imshow("Rectangle", image)
        #cv2.imshow("All Rectangles", image_all)

        # Draw and wipe the rectangles
        cv2.drawContours(img, rect_contours, -1, (255, 255, 255), 5)
        
        #TODO remove after testing
        # Display the result
        #cv2.imshow("Processed Image", img)
        
        print(f"Detected {len(rect_contours)} rectangles")
        return img, all_rect_areas