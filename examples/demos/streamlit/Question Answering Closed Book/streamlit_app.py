############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os
import pandas as pd
import numpy as np

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
    
spark = sparknlp.start()

def create_pipeline(model):
    document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("documents")

    sentence_detector = SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en")\
        .setInputCols(["documents"])\
        .setOutputCol("questions")
        
    t5 = T5Transformer()\
        .pretrained("google_t5_small_ssm_nq")\
        .setTask('trivia question:')\
        .setInputCols(["questions"])\
        .setOutputCol("answers")
        
    qa_pp = Pipeline(stages=[document_assembler, sentence_detector, t5])

    return qa_pp

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['answers'][0].result

############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Automatically Answer Questions", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.title("Automatically Answer Questions (Closed Book)")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["google_t5_small_ssm_nq"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/QUESTION_ANSWERING_CLOSED_BOOK.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Who is Clark Kent?",
    "Which is the capital of Bulgaria ?",
    "Which country tops the annual global democracy index compiled by the economist intelligence unit?",
    "In which city is the Eiffel Tower located?",
    "Who is the founder of Microsoft?"
]

st.subheader("Automatically generate answers to questions without context")

selected_text = st.selectbox('Select an Example:', examples)
custom_input = st.text_input('Try it yourself!')

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader('Selected Text')
st.write(selected_text)

Pipeline = create_pipeline(model)
output = fit_data(Pipeline, selected_text)

st.subheader('Prediction')
st.write(output)