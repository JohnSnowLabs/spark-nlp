############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()

@st.cache_resource
def create_pipeline(model, labels):

    image_assembler = ImageAssembler() \
        .setInputCol("image") \
        .setOutputCol("image_assembler")

    imageClassifier = CLIPForZeroShotClassification \
        .pretrained() \
        .setInputCols(["image_assembler"]) \
        .setOutputCol("label") \
        .setCandidateLabels(labels)

    pipeline = Pipeline(stages=[
        image_assembler,
        imageClassifier,
    ])

    return pipeline

def fit_data(pipeline, data):
  
    model = pipeline.fit(data)
    light_pipeline = LightPipeline(model)
    annotations_result = light_pipeline.fullAnnotateImage(data)
    pharsed_result = (lambda data: data[data.find("a photo of a"):].split(",")[0] if "a photo of a" in data else "Not found")(str(annotations_result[0]['label']))
    return pharsed_result

def save_uploadedfile(uploadedfile):
    filepath = os.path.join(IMAGE_FILE_PATH, uploadedfile.name)
    with open(filepath, "wb") as f:
        if hasattr(uploadedfile, 'getbuffer'):
            f.write(uploadedfile.getbuffer())
        else:
            f.write(uploadedfile.read())
        
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="CLIPForZeroShotClassification for Zero Shot Classification", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("CLIPForZeroShotClassification for Zero Shot Classification")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)
model = st.sidebar.selectbox("Choose the pretrained model", ['CLIPForZeroShotClassification'], help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link = """<a href="https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/CLIPForZeroShotClassification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

IMAGE_FILE_PATH = "/content/CLIPForZeroShotClassification/input"
image_files = sorted([file for file in os.listdir(IMAGE_FILE_PATH) if file.split('.')[-1]=='png' or file.split('.')[-1]=='jpg' or file.split('.')[-1]=='JPEG' or file.split('.')[-1]=='jpeg'])

st.subheader("Summarize text to make it shorter while retaining meaning.")

img_options = st.selectbox("Select an image", image_files)
uploadedfile = st.file_uploader("Try it for yourself!")

if uploadedfile:
    file_details = {"FileName":uploadedfile.name,"FileType":uploadedfile.type}
    save_uploadedfile(uploadedfile)
    selected_image = f"{IMAGE_FILE_PATH}/{uploadedfile.name}"
elif img_options:
    selected_image = f"{IMAGE_FILE_PATH}/{img_options}"

candidateLabels = [
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a hen",
    "a photo of a hippo",
    "a photo of a room",
    "a photo of a tractor",
    "a photo of an ostrich",
    "a photo of an ox"]

lables = st.multiselect("Select labels", candidateLabels, default=candidateLabels)

st.subheader('Classified Image')

image_size = st.slider('Image Size', 400, 1000, value=400, step = 100)

try:
    st.image(f"{IMAGE_FILE_PATH}/{selected_image}", width=image_size)
except:
    st.image(selected_image, width=image_size)

st.subheader('Classification')

Pipeline = create_pipeline(model, lables)
output = fit_data(Pipeline, selected_image)

st.markdown(f'This document has been classified as  : **{output}**')
