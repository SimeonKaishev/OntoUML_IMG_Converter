import uuid
import numpy as np
from rdfltpib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
from tensorflow.keras.preprocessing import image
from utils.img_utils import imgUtils


class OntologyUtils:
    """
    A utility class for creating and managing ontology-related operations.

    Methods
    -------
    create_class_triples(class_list, g, namespace)
        Creates and adds class triples to the RDF graph.
    
    create_relationship_triples(relations, g, namespace, class_uris)
        Creates and adds relationship triples to the RDF graph.
    
    save_as_ontouml_vocab(classes, class_relations, diagram_name)
        Saves the given classes and relations as an OntoUML vocabulary in a Turtle file.
    """

    @staticmethod
    def create_class_triples(class_list, g, namespace):
        """
        Creates and adds class triples to the RDF graph.

        Parameters
        ----------
        class_list : list
            A list of UMLClass objects to add to the RDF graph.
        g : rdflib.Graph
            The RDF graph to which the triples will be added.
        namespace : rdflib.Namespace
            The namespace to use for the triples.

        Returns
        -------
        dict
            A dictionary mapping class names to their corresponding URIs.
        """
        class_uris = {}
        for class_item in class_list:
            class_id = str(uuid.uuid4())
            class_item.setId(class_id)
            class_name = class_item.name
            class_stereotype = class_item.stereotype
            
            class_uri = URIRef(f"{namespace}{class_id}")
            class_uris[class_name] = class_uri  # Store URI for later reference

            # Create triples
            g.add((class_uri, RDF.type, namespace.Class))
            g.add((class_uri, namespace.name, Literal(class_name, lang="en")))
            if class_stereotype != "":
                g.add((class_uri, namespace.stereotype, URIRef(f"{namespace}{class_stereotype}")))
        
        return class_uris

    @staticmethod
    def create_relationship_triples(relations, g, namespace, class_uris):
        """
        Creates and adds relationship triples to the RDF graph.

        Parameters
        ----------
        relations : list
            A list of Relation objects to add to the RDF graph.
        g : rdflib.Graph
            The RDF graph to which the triples will be added.
        namespace : rdflib.Namespace
            The namespace to use for the triples.
        class_uris : dict
            A dictionary mapping class names to their corresponding URIs.
        """
        for relation in relations:
            rel_id = str(uuid.uuid4())
            rel_uri = URIRef(f"{namespace}{rel_id}")
            if relation.rel_type == "Generalisation" and relation.source != -1:
                # Add relationship
                g.add((rel_uri, RDF.type, namespace.Generalization))
                # Add relationship classes
                for index, cls in enumerate(relation.classes):
                    if index == relation.source:
                        # Add as general
                        g.add((rel_uri, namespace.general, class_uris[cls.name]))
                    else:
                        # Add as specific
                        g.add((rel_uri, namespace.specific, class_uris[cls.name]))
            else:
                continue

    @staticmethod
    def save_as_ontouml_vocab(classes, class_relations, diagram_name):
        """
        Saves the given classes and relations as an OntoUML vocabulary in a Turtle file.

        Parameters
        ----------
        classes : list
            A list of UMLClass objects to save.
        class_relations : list
            A list of Relation objects to save.
        diagram_name : str
            The name of the diagram to use for the Turtle file.
        """
        # Create a new graph
        g = Graph()

        # Define the namespaces
        ONTOUML = Namespace("http://example.org/ontouml#")

        # Bind the namespace prefix to the graph
        g.bind("ontouml", ONTOUML)

        # Create and add class and relationship triples to the graph
        uris = OntologyUtils.create_class_triples(classes, g, ONTOUML)
        OntologyUtils.create_relationship_triples(class_relations, g, ONTOUML, uris)
        
        # Save the graph to a Turtle file
        g.serialize(destination=f"{diagram_name}.ttl", format="turtle")


    def classify_image(model, img):
        """
        Classifies an image into OntoUML or not using a pre-trained model.

        Parameters
        ----------
        model : tensorflow.keras.Model
            The pre-trained model used for classification.
        img : numpy.ndarray
            Image to classify

        Returns
        -------
        bool
            True if the image was classified as the first class (index 0), False otherwise.
        """
        # Load the image
        img = imgUtils.resize_and_pad_image(img)

        # Convert the image to a numpy array and expand dimensions to fit the model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize the image array if necessary (assuming the model was trained with normalized inputs)
        img_array /= 255.0
        
        # Use the model to predict the class of the image
        predictions = model.predict(img_array)
        
        # Assuming the model output is a probability for each class
        # and that class 0 or the first class is the desired class
        class_index = np.argmax(predictions[0])
        
        # Return True if the image was classified as the first class (0), False otherwise
        return class_index == 0