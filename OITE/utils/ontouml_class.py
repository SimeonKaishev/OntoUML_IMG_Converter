from .rectangle import Rectangle
import cv2
import pytesseract
from .img_utils import imgUtils
from .text_utils import TextUtils
import Levenshtein

class OntoUMLClass:
    """
    A class used to represent a UML Class diagram component.

    Attributes
    ----------
    list : list
        A list of rectangles associated with the UML class.
    top : Rectangle
        The top rectangle in the UML class.
    mid : Rectangle
        The middle rectangle in the UML class.
    bottom : Rectangle
        The bottom rectangle in the UML class.
    temp_rect : Rectangle
        A temporary rectangle used for size calculations.
    whole : Rectangle
        The whole rectangle representing the UML class's size.
    title : str
        The title of the UML class.
    attris_str : str
        The attributes string of the UML class.
    methods_str : str
        The methods string of the UML class.
    name : str
        The name of the UML class.
    stereotype : str
        The stereotype of the UML class.
    id : str
        The identifier of the UML class.
    """
    def __init__(self):
        self.list = []
        self.top = None
        self.mid = None
        self.bottom = None
        self.temp_rect = None
        self.whole = None
        self.title = ""
        self.attris_str = ""
        self.methods_str = ""
        self.name = ""
        self.stereotype = ""
        self.id = ""

    # All OntoUML stereotypes
    stereotypes = [
    "abstract", "category", "collective", "datatype", "enumeration",
    "event", "historicalRole", "historicalRoleMixin", "kind", "mixin",
    "mode", "phase", "phaseMixin", "quality", "quantity", "relator",
    "role", "roleMixin", "situation", "subkind", "type", "powertype", 
    "highordertype", "hou", "universal", "2ndOT", "relatorKind", "modeKind", 
    "quantityKind", "collectiveKind", "qualityKind"
    ]

    # Depricated stereotypes
    stereotypes_old = ["powertype", "highordertype", "hou", "universal", 
                "2ndOT", "relatorKind", "modeKind", "quantityKind",
                "collectiveKind", "qualityKind"]
    
    # New stereotypes to replace depricated ones with
    stereotypes_new = ["type","type","type","type","type","relator", "mode",
                "quantity", "collective", "quality"]

    def setTitle(self, title):
        """
        Sets the title of the UML class.

        Parameters
        ----------
        title : str
            The title to set.
        """
        self.title = title.strip()

    def setAttrisStr(self, attris_str):
        """
        Sets the attributes string of the UML class.

        Parameters
        ----------
        attris_str : str
            The attributes string to set.
        """
        self.attris_str = attris_str.strip()

    def setMethodsStr(self, methods_str):
        """
        Sets the methods string of the UML class.

        Parameters
        ----------
        methods_str : str
            The methods string to set.
        """
        self.methods_str = methods_str.strip()

    def setName(self, name_str):
        """
        Sets the name of the UML class.

        Parameters
        ----------
        name_str : str
            The name to set.
        """
        self.name = name_str

    def setStereotype(self, stereotype_str):
        """
        Sets the stereotype of the UML class.

        Parameters
        ----------
        stereotype_str : str
            The stereotype to set.
        """
        self.stereotype = stereotype_str
    
    def setId(self, id):
        """
        Sets the identifier of the UML class.

        Parameters
        ----------
        id : str
            The identifier to set.
        """
        self.id = id

    @staticmethod
    def merge_into_class(all_rect_areas_in_cd):
        """
        Merges rectangles into UML class objects based on spatial relationships.

        Parameters
        ----------
        all_rect_areas_in_cd : list
            A list containing information about detected rectangles.

        Returns
        -------
        list
            A list of OntoUMLClass objects created from the detected rectangles.
        """
        result = []

        # Get all rectangles
        rect_area_list = all_rect_areas_in_cd[1]

        for all_rect_index in range(len(rect_area_list)):
            current_rect = rect_area_list[all_rect_index]

            # If the current rect already belongs to a class, skip it
            if current_rect.within_cls_obj:
                continue

            # Otherwise, assign it to a new class
            uml_class = OntoUMLClass()
            uml_class.list.append(current_rect)
            current_rect.within_cls_obj = True

            # This temp_rect is for updating the class's size
            uml_class.temp_rect = current_rect.clone()

            # Compare the temp_rect with other rects in all_rect_areas
            for j in range(all_rect_index + 1, len(rect_area_list)):
                other_rect = rect_area_list[j]

                # Check if the horizontal distance between left-tops is within 3-5 pixels
                if abs(other_rect.tl()[0] - uml_class.temp_rect.tl()[0]) <= 5:
                    if uml_class.temp_rect.tl()[1] < other_rect.tl()[1]:
                        if other_rect.tl()[1] - uml_class.temp_rect.tl()[1] - uml_class.temp_rect.height <= 5:
                            uml_class.list.append(other_rect)
                            other_rect.within_cls_obj = True
                            uml_class.temp_rect.height += other_rect.height
                    else:
                        if uml_class.temp_rect.tl()[1] - other_rect.tl()[1] - other_rect.height <= 5:
                            uml_class.list.append(other_rect)
                            other_rect.within_cls_obj = True
                            uml_class.temp_rect.x = other_rect.x
                            uml_class.temp_rect.y = other_rect.y
                            uml_class.temp_rect.height += other_rect.height

            # Sort the rects in the class into top, mid, and bottom
            uml_class.list.sort(key=lambda r: r.y)

            for j, r_in_l in enumerate(uml_class.list):
                if r_in_l is not None:
                    if j == 0:
                        uml_class.top = r_in_l
                    elif j == 1:
                        uml_class.mid = r_in_l
                    elif j == 2:
                        uml_class.bottom = r_in_l

            # Adjust top, mid, bottom if necessary
            if uml_class.mid:
                if uml_class.bottom:
                    if uml_class.top.y + uml_class.top.height >= uml_class.bottom.y + uml_class.bottom.height:
                        top_old_height = uml_class.top.height
                        uml_class.top = uml_class.mid.clone()
                        uml_class.mid = uml_class.bottom.clone()
                        uml_class.bottom.y = uml_class.top.y + uml_class.top.height + uml_class.mid.height
                        uml_class.bottom.height = top_old_height - uml_class.top.height - uml_class.mid.height
                else:
                    if uml_class.top.y + uml_class.top.height >= uml_class.mid.y + uml_class.mid.height:
                        top_old_height = uml_class.top.height
                        uml_class.top = uml_class.mid.clone()
                        uml_class.mid.y = uml_class.top.y + uml_class.top.height
                        uml_class.mid.height = top_old_height - uml_class.top.height

            # Calculate the whole class's size
            whole_height = uml_class.top.height
            if uml_class.mid:
                whole_height += uml_class.mid.height
            if uml_class.bottom:
                whole_height += uml_class.bottom.height

            uml_class.whole = Rectangle(None, None, None, uml_class.top.x, uml_class.top.y, uml_class.top.width, whole_height)
            
            result.append(uml_class)

        return result
    
    @staticmethod
    def detectText(cls_diagram, classes):
        """
        Detects and extracts text from the UML class diagram's regions and assigns it to the appropriate attributes.

        Parameters
        ----------
        cls_diagram : numpy.ndarray
            The image of the UML class diagram.
        classes : list
            A list of OntoUMLClass objects to process.

        Returns
        -------
        list
            The list of OntoUMLClass objects with text extracted and set.
        """
        for uc in classes:
            try:
                # Process the top region
                if uc.top is not None:
                    img = imgUtils.cutImage(cls_diagram, uc.top)
                    # Increase image size
                    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    if img.size > 0:
                        img_string = pytesseract.image_to_string(img)
                        uc.setTitle(img_string)
                        print(img_string)
                        print("\n")

                # Process the mid region
                if uc.mid is not None:
                    img = imgUtils.cutImage(cls_diagram, uc.mid)
                    if img.size > 0:
                        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        uc.setAttrisStr(pytesseract.image_to_string(img, lang='eng'))

                # Process the bottom region
                if uc.bottom is not None:
                    if uc.bottom.y > uc.whole.y:
                        uc.bottom.y = uc.whole.y

                    img = imgUtils.cutImage(cls_diagram, uc.bottom)
                    if img.size > 0:
                        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        uc.setMethodsStr(pytesseract.image_to_string(img, lang='eng'))

            except Exception as e:
                print(f"Error processing class {uc}: {e}")

        return classes
    

    @staticmethod
    def set_names_and_remove_empty_classes(classes):
        """
        Sets the names and stereotypes for UML classes and removes empty classes.

        Parameters
        ----------
        classes : list
            A list of OntoUMLClass objects to process.

        Returns
        -------
        list
            The updated list of OntoUMLClass objects with names and stereotypes set and empty classes removed.
        """
        classes_to_remove = []
        for uml_class in classes:
            stereotype, class_name = TextUtils.extract_class_info(uml_class.title)
            if class_name == "" and stereotype == "":
                classes_to_remove.append(uml_class)
            else:
                uml_class.setName(class_name)
                uml_class.setStereotype(stereotype)
                print(f"class: {class_name}, stereotype: {stereotype}")

        # Remove classes from classes_to_remove
        for cls in classes_to_remove:
            classes.remove(cls)

        return classes
    
    @staticmethod
    def update_stereotype(stereotype):
        """
        Update old OntoUML stereotypes to the corresponding new ones.

        Parameters
        ----------
        stereotype : str
            The old stereotype to update.

        Returns
        -------
        str
            The updated stereotype if found in the old stereotypes list, otherwise returns the original stereotype.
        """
        if stereotype in OntoUMLClass.stereotypes_old:
            index = OntoUMLClass.stereotypes_old.index(stereotype)
            return OntoUMLClass.stereotypes_new[index]
        else:
            return stereotype

    @staticmethod
    def check_stereotypes(classes):
        """
        Checks and updates stereotypes of the given UML classes, counting OntoUML stereotypes.

        Parameters
        ----------
        classes : list
            A list of UMLClass objects to process.

        Returns
        -------
        int
            The number of OntoUML stereotypes found and updated.
        """
        num_ontouml_stereotypes = 0
        for cls in classes:
            # Get stereotype
            stereotype = cls.stereotype
            # Get the distance to the closest OntoUML stereotype
            similarities = [(term, Levenshtein.distance(stereotype, term)) for term in OntoUMLClass.stereotypes]
            # Find the term with the minimum distance
            most_similar_term, min_distance = min(similarities, key=lambda x: x[1])
            # If distance <= 2, save it and increase num_ontouml_stereotypes
            if min_distance <= 2:
                num_ontouml_stereotypes += 1
                stereotype_new = OntoUMLClass.update_stereotype(most_similar_term)
                cls.stereotype = stereotype_new
        return num_ontouml_stereotypes