############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os
import random
import base64
import re

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import *
import pyspark.sql.functions as F

spark = sparknlp.start()

@st.cache_resource
def create_pipeline(model, language):
    
  documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

  tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

  stop_words = StopWordsCleaner.pretrained(model, language) \
    .setInputCols(["token"]) \
    .setOutputCol("cleanTokens")

  pipeline = Pipeline(stages=[documentAssembler, tokenizer, stop_words])
  return pipeline

def fit_data(pipeline, data):
  empty_df = spark.createDataFrame([['']]).toDF('text')
  pipeline_model = pipeline.fit(empty_df)
  model = LightPipeline(pipeline_model)
  result = model.annotate(data)

  return result
        
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Stopwords for East Asian", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Stopwords Remover for East Asian")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

model_mappings = {
    "Indonesian": ["stopwords_iso", "id"],
    "Japanese": ["stopwords_iso","ja"],
    "Korean": ["stopwords_iso","ko"],
    "Tagalog": ["stopwords_iso", "tl"],
    "Vietnamese": ["stopwords_iso", "vi"],
    "Chinese": ["stopwords_iso", "zh"],
}

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)
model = st.sidebar.selectbox("Choose the pretrained model", list(model_mappings.keys()), help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link = """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

folder_path = f"/content/StopWords East Asain/inputs/{model}"
examples = [lines[1].strip() for filename in os.listdir(folder_path) if filename.endswith('.txt') for lines in [open(os.path.join(folder_path, filename), 'r').readlines()] if len(lines) >= 2]

st.subheader(f"This model removes stopwords from {model} documents.")

selected_text = st.selectbox("Select an example", examples)
custom_input = st.text_input("Try it for yourself!")

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader("Selected Text:")
st.write(selected_text)

st.subheader(f"Filtered Text:")
Pipeline = create_pipeline(model_mappings[model][0] , model_mappings[model][1])
output = fit_data(Pipeline, selected_text)

filtered_sentence = ' '.join(output["cleanTokens"])

st.sidebar.write("")
st.markdown(f"**{filtered_sentence}**")

stop_words_removed = list((set(output["token"]) | set(output["cleanTokens"])) - (set(output["token"]) & set(output["cleanTokens"])))

table, table2, table3 = st.columns(3)
with table:
    st.subheader('Basic Tokens:')
    st.table(output["token"])   
with table2:
    st.subheader('Tokens after removing Stop Words:')
    st.table(output["cleanTokens"]) 
with table3:
    st.subheader('Stop Words Removed:')
    st.table(stop_words_removed) 

