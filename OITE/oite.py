import cv2
from utils.img_utils import imgUtils
from utils.rectangle import Rectangle
from utils.ontouml_class import OntoUMLClass
from utils.relation import Relation
from utils.ontology_utils import OntologyUtils

#TODO implement

# load model for img classification
model_OntoUML_detect = None
# load model for rel classification
model_relationship_detect = None

#load image
image_path = 'offline.png'
diagram_name = image_path.replace(".png", "")

processed_img, binary_img = imgUtils.get_img_processed(image_path)
clean_img = cv2.imread(image_path)

# check if image is ontoUML with model

# get rectangles
no_class_img, rectangles = Rectangle.detect_rectangles(processed_img, image_path)
no_class_img2, rectangles2 =  Rectangle.detect_rectangles(binary_img, image_path)
binary = False

# take whichever processing resulted in more rectangles detected
if len(rectangles2) > len(rectangles):
  rectangles, no_class_img = rectangles2, no_class_img2
  binary = True

# get classes
all_rect_areas_in_cd = (processed_img, rectangles)
merged_classes = OntoUMLClass.merge_into_class(all_rect_areas_in_cd)
print(len(merged_classes))

# extract elements from class text
classes = OntoUMLClass.detectText(processed_img, merged_classes)
classes = OntoUMLClass.set_names_and_remove_empty_classes(classes)

# check stereotypes for OntoUML
ontouml_stereotypes = OntoUMLClass.check_stereotypes(classes)
if ontouml_stereotypes < 1:
  # TODO not ontoUML, so dont continue converting
  pass

# wipe class rectangles again, incese extra objects were detected during rectangle detection
if binary:
  no_class_img2 = imgUtils.wipe_classes(binary_img, classes)
else:
  no_class_img2 = imgUtils.wipe_classes(processed_img, classes)
cv2.imshow(no_class_img2)

# get relationships
remaining_shapes = Relation.detect_relations(no_class_img, image_path)
class_relations = Relation.get_relations(remaining_shapes, classes)

remaining_shapes = Relation.detect_relations(no_class_img2, image_path)
class_relations2 = Relation.get_relations(remaining_shapes, classes)

# take whichever detection found the most relationships
if len(class_relations) <= len(class_relations2):
  # If equal # of relations, pick the one which connects more classes
  if len(class_relations) == len(class_relations2):
    cl1 = Relation.get_number_of_classes(class_relations)
    cl2 = Relation.get_number_of_classes(class_relations2)
    if cl1 <= cl2:
      class_relations = class_relations
  else:
    class_relations = class_relations2

class_relations = Relation.get_relation_types(class_relations, model_relationship_detect, processed_img, no_class_img)

# save to ttl
OntologyUtils.save_as_ontouml_vocab(classes, class_relations, diagram_name)
