import cv2
import numpy as np

class Relation:
    """
    A class used to represent a relation in a UML diagram.

    Attributes
    ----------
    classes : list
        A list of UMLClass objects involved in the relation.
    endpoints : list
        A list of endpoints associated with the relation.
    curve : any
        The curve representing the relation.
    rel_type : str
        The type of the relation.
    source : int
        The source of the relation.
    """

    def __init__(self, curve):
        """
        Constructs all the necessary attributes for the Relation object.

        Parameters
        ----------
        curve : any
            The curve representing the relation.
        """
        self.classes = []
        self.endpoints = []
        self.curve = curve
        self.rel_type = ""
        self.source = -1
        
    def addClass(self, uml_class, endpoint):
        """
        Adds a UML class and its corresponding endpoint to the relation.

        Parameters
        ----------
        uml_class : UMLClass
            The UML class to add to the relation.
        endpoint : any
            The endpoint associated with the UML class.
        """
        if uml_class not in self.classes:
            self.classes.append(uml_class)
            self.endpoints.append(endpoint)

    def set_type(self, rel_type, source=-1):
        """
        Sets the type and source of the relation.

        Parameters
        ----------
        rel_type : str
            The type of the relation.
        source : int, optional
            The source of the relation (default is -1).
        """
        self.rel_type = rel_type
        self.source = source

    # TODO rename to smth that makes sence
    @staticmethod
    def detect_relations(original_img, image_path):
        """
        Detects possible relations in an image by finding and approximating contours.

        Parameters
        ----------
        original_img : numpy.ndarray
            The original image in which to detect relations.
        image_path : str
            The path to the image file.

        Returns
        -------
        list
            A list of approximated curves representing the detected relations.
        """
        img = original_img.copy()
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        relations = []

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
            relations.append(approx_curve)     

        return relations
    


    @staticmethod
    def get_relations(relations, classes):
        """
        Identifies and returns the relations involving multiple UML classes.

        Parameters
        ----------
        relations : list
            A list of detected relations.
        classes : list
            A list of UMLClass objects.

        Returns
        -------
        list
            A list of Relation objects that involve multiple UML classes.
        """
        final_relations = []
        class_prox = 15 

        for relation in relations:
            new_relation = Relation(relation)
            related_classes = []  
            points = []
            for p in relation:
                new_point = [p[0][0], p[0][1]]
                points.append(new_point)
            points = np.array(points)
            for point in points:
                point_up = [point[0], point[1] + class_prox]
                point_ur = [point[0] + class_prox, point[1] + class_prox]
                point_ul = [point[0] + class_prox, point[1] - class_prox]
                point_down = [point[0], point[1] - class_prox]
                point_dr = [point[0] - class_prox, point[1] + class_prox]
                point_dl = [point[0] - class_prox, point[1] - class_prox]
                point_left = [point[0] - class_prox, point[1]]
                point_right = [point[0] + class_prox, point[1]]
                for uml_class in classes:
                    if (uml_class.whole.contains(point_up) or uml_class.whole.contains(point_down) or 
                        uml_class.whole.contains(point_left) or uml_class.whole.contains(point_right)):
                        related_classes.append(uml_class)
                        new_relation.addClass(uml_class, point)
                    if (uml_class.whole.contains(point_ul) or uml_class.whole.contains(point_ur) or 
                        uml_class.whole.contains(point_dl) or uml_class.whole.contains(point_dr)):
                        related_classes.append(uml_class)
                        new_relation.addClass(uml_class, point)
            # Only if it has > 1 class
            if len(new_relation.classes) >= 2:
                final_relations.append(new_relation)

        return final_relations

    @staticmethod
    def printRelations(relations):
        """
        Prints and displays the detected relations and their associated UML classes.

        Parameters
        ----------
        relations : list
            A list of Relation objects to print and display.
        """
        image_path = "path/to/your/image"  # Replace with actual image path
        image_all = cv2.imread(image_path)
        for rel in relations:
            image = cv2.imread(image_path)
            cv2.polylines(image, [rel.curve.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(image_all, [rel.curve.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
            for ep in rel.endpoints:
                cv2.circle(image, (int(ep[0]), int(ep[1])), 3, (255, 0, 255), -1)
                cv2.circle(image_all, (int(ep[0]), int(ep[1])), 3, (255, 0, 255), -1)
            #cv2.imshow("Relation", image)
            print("Related classes:")
            for rc in rel.classes:
                print(rc.name)
            print("\n")
        cv2.imshow("All Relations", image_all)
        
    @staticmethod
    def get_number_of_classes(class_relations):
        """
        Returns the total number of UML classes involved in the given relations.

        Parameters
        ----------
        class_relations : list
            A list of Relation objects.

        Returns
        -------
        int
            The total number of UML classes involved in the given relations.
        """
        clss = 0
        for rel in class_relations:
            clss += len(rel.classes)
        return clss
    

    @staticmethod
    def calculate_square_size(image_size, num_classes, scale_factor=5):
        """
        Calculate the size of the square based on the image size and number of classes.
        
        Parameters
        ----------
        image_size : tuple
            The size of the image (width, height).
        num_classes : int
            The number of classes.
        scale_factor : float, optional
            A scaling factor to adjust the size calculation (default is 5).
        
        Returns
        -------
        int
            The calculated size of the square.
        """
        width, height = image_size
        # Calculate the diagonal length of the image
        image_diagonal = (width**2 + height**2)**0.5
        # Determine the base size proportion using the scaling factor
        base_proportion = image_diagonal / scale_factor
        # Calculate the size of the square, inversely proportional to the square root of the number of classes
        square_size = base_proportion / (num_classes**0.5)
        # Ensure the square size is at least 1 pixel
        square_size = max(1, int(square_size))
        return square_size

    @staticmethod
    def classify_relation_sign(image, model, input_size=(100, 100)):
        """
        Classify the relation sign in the given image using a machine learning model.

        Parameters
        ----------
        image : numpy.ndarray
            The image containing the relation sign.
        model : keras.Model
            The machine learning model to use for classification.
        input_size : tuple, optional
            The input size expected by the model (default is (100, 100)).

        Returns
        -------
        int
            The predicted class of the relation sign.
        """
        # Resize to the image size the model takes
        image_resized = cv2.resize(image, input_size)
        cv2.imshow("Resized Image", image_resized)
        # Convert to RGB
        rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        # Add batch dimension
        image_batch = np.expand_dims(rgb_image, axis=0)
        # Predict the class probabilities
        predictions = model.predict(image_batch)
        print(predictions)
        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class

    @staticmethod
    def get_relation_type(point, image, size, model):
        """
        Determine the type of the relation at the given point in the image using the model.

        Parameters
        ----------
        point : tuple
            The coordinates of the point (x, y).
        image : numpy.ndarray
            The image containing the relation.
        size : int
            The size of the square around the point to crop.
        model : keras.Model
            The machine learning model to use for classification.

        Returns
        -------
        str
            The type of the relation ("Generalisation" or "Other").
        """
        point_x, point_y = int(point[0]), int(point[1])
        # Calculate the coordinates for cropping
        top_left_x = max(point_x - size // 2, 0)
        top_left_y = max(point_y - size // 2, 0)
        bottom_right_x = min(point_x + size // 2, image.shape[1])
        bottom_right_y = min(point_y + size // 2, image.shape[0])
        # Crop the image
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cropped_image = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        sign = Relation.classify_relation_sign(cropped_image, model)
        if sign[0] == 0:
            print("Generalisation")
            return "Generalisation"
        else:
            return "Other"

    @staticmethod
    def detect_triangles(image):
        """
        Detect triangles in the given image.

        Parameters
        ----------
        image : numpy.ndarray
            The image in which to detect triangles.

        Returns
        -------
        bool
            True if triangles are detected, False otherwise.
        """
        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Edge detection using Canny
        edged = cv2.Canny(blurred, 60, 150)
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        triangles = []
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Check if the approximated contour has 3 vertices
            if len(approx) == 3:
                triangles.append(approx)
        if len(triangles) > 0:
            return True
        return False

    @staticmethod
    def get_relation_types(class_relations, model, classes, processed_img, no_class_img):
        """
        Determine the types of relations for the given class relations using the model.

        Parameters
        ----------
        class_relations : list
            A list of Relation objects.
        model : keras.Model
            The machine learning model to use for classification.
        classes : list
            List of OntoUMLClass objects representing the classes in the diagram
        processed_img :  numpy.ndarray
            image of the diagram, gray scaled, sharpened
        no_class_img :  numpy.ndarray
            image of the diagram, with the classes wiped off

        Returns
        -------
        list
            The updated list of Relation objects with their types set.
        """
        height, width = processed_img.shape[:2]
        image_size = (width, height)
        sq_size = Relation.calculate_square_size(image_size, len(classes))
        for relation in class_relations:
            if len(relation.classes) >= 2:
                for index, ep in enumerate(relation.endpoints):
                    ep_type = Relation.get_relation_type(ep, no_class_img, sq_size, model)
                    if ep_type == "Generalisation":
                        relation.set_type(ep_type, index)
                if relation.rel_type == "":
                    relation.set_type("Other")
        return class_relations