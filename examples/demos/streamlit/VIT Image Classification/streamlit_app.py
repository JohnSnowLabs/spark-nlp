############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os
import re

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()

def create_pipeline(model):

    image_assembler = ImageAssembler() \
                .setInputCol("image") \
                .setOutputCol("image_assembler")

    image_classifier = ViTForImageClassification \
        .pretrained(model) \
        .setInputCols("image_assembler") \
        .setOutputCol("class")

    pipeline = Pipeline(stages=[
        image_assembler,
        image_classifier,
    ])

    return pipeline

def fit_data(pipeline, data):

    empty_df = spark.createDataFrame([['']]).toDF('text')
    #data_df = spark.read.format("image").option("dropInvalid", value = True).load(path=data)
    model = pipeline.fit(empty_df)
    light_pipeline = LightPipeline(model)
    annotations_result = light_pipeline.fullAnnotateImage(data)
    pharsd_res = re.search(r"Annotation\(category, \d+, \d+, (\w+),", str(annotations_result[0]['class']))
    result = pharsd_res.group(1) if pharsd_res else "No category found"
    return result

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
    layout="wide", page_title="ViT for Image Classification", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("ViT for Image Classification")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list =   ['image_classifier_vit_base_cats_vs_dogs', 'image_classifier_vit_base_patch16_224', 'image_classifier_vit_CarViT', 'image_classifier_vit_base_beans_demo', 'image_classifier_vit_base_food101', 'image_classifier_vit_base_patch16_224_in21k_finetuned_cifar10']
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link = """<a href="https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/ViTForImageClassification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

IMAGE_FILE_PATH = f"/content/VIT Image Classification/inputs/{model}"
image_files = sorted([file for file in os.listdir(IMAGE_FILE_PATH) if file.split('.')[-1]=='png' or file.split('.')[-1]=='jpg' or file.split('.')[-1]=='JPEG' or file.split('.')[-1]=='jpeg'])

st.subheader("This model identifies image classes using the vision transformer (ViT).")

img_options = st.selectbox("Select an image", image_files)
uploadedfile = st.file_uploader("Try it for yourself!")

if uploadedfile:
    file_details = {"FileName":uploadedfile.name,"FileType":uploadedfile.type}
    save_uploadedfile(uploadedfile)
    selected_image = f"{IMAGE_FILE_PATH}/{uploadedfile.name}"
elif img_options:
    selected_image = f"{IMAGE_FILE_PATH}/{img_options}"

st.subheader('Classified Image')

image_size = st.slider('Image Size', 400, 1000, value=400, step = 100)

try:
    st.image(f"{IMAGE_FILE_PATH}/{selected_image}", width=image_size)
except:
    st.image(selected_image, width=image_size)

st.subheader('Classification')

Pipeline = create_pipeline(model)
output = fit_data(Pipeline, selected_image)

st.markdown(f'This document has been classified as  : **{output}**')