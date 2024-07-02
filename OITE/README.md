# OntoUML Image Taxonomy Extractor (OITE)
## System description
This system performs batch conversion of OntoUML diagram images into OntoUML Vocabulary from a user-specified directory. It processes each image, and if it classifies the image as OntoUML, it translates it into OntoUML Vocabulary using image processing and machine learning models. The resulting diagrams are stored as TTL files in the output folder.

## Installation instructions
### Clone the repository
To get started, clone the repository to your local machine using Git. Open your terminal or Git Bash and run the following command:
   ```sh
    git clone https://github.com/SimeonKaishev/OntoUML_IMG_Converter.git
   ```
This command will download the project files to a directory named OntoUML_IMG_Converter.
### Install Tesseract-OCR
To run this project, you need to install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

To do so please follow the instructions in the linked GitHub repository, and do not forget to add resseract to your path variable!

### Download AI models

To run this project, you need to download the pre-trained AI models from Hugging Face.

#### Step-by-Step Guide

1. **Download the model files from Hugging Face**

   - Visit the Hugging Face repository: [OITE Models on Hugging Face](https://huggingface.co/sskaishev/OITE_Models)
   - Download the following folders:
     - `my_model_3 class_good`
     - `Relationship_classifier`

2. **Place the downloaded folders in the `OITE/models/` directory of the project**

### Install required python libraries
To install the necessary Python libraries for this project, follow these steps:

1. Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. Open a terminal or command prompt.

3. Navigate to the directory where your `requirements.txt` file is located. For example:
   ```sh
   cd /path_to_project/OntoUML_IMG_Converter/OITE
   ```
4. Use the following command to install the required libraries:
   ```sh
    pip install -r requirements.txt
   ```
This command will read the requirements.txt file and install all the listed libraries.


## Usage instructions
### 1. Prepare the Images:
Ensure that all the images you want to convert are in the same directory.
### 2. Navigate to the Project Directory:
Open your terminal or command prompt and change to the project directory:
   ```sh
   cd /path_to_project/OntoUML_IMG_Converter/OITE
   ```
### 3. Run the Script:
Execute the main script:
   ```sh
   python oite.py
   ```
### 4. Let the Models Load:
The AI models used by the system will take some time to load when the script is run, so be patient during this process.
### 5. Provide the Absolute Path:
When prompted, enter the absolute path to the folder containing the images. The application will then process the images and translate those it classifies as OntoUML.

After providing the path, the system will analyze the images, classify those it identifies as OntoUML, and convert them into OntoUML vocabulary. The results will be stored as TTL files in the output folder.