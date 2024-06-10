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
def create_pipeline(model):
    documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("documents")

    t5 = T5Transformer.pretrained(model) \
      .setTask("gec:") \
      .setInputCols(["documents"]) \
      .setMaxOutputLength(200) \
      .setOutputCol("corrections")

    pipeline = Pipeline().setStages([documentAssembler, t5])
    return pipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['corrections'][0].result

############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Sentence Grammar", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.title("Evaluate Sentence Grammar")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["t5_grammar_error_corrector"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Anna and Mike is going skiing and they is liked is",
    "Anna and Mike like to dance",
    "the pond froze solid .",
    "they made him to exhaustion .",
    "bill floated into the cave .",
    "the parcel reached john .",
    "john tries to meet often mary .",
    "whose book did you find ?"
]

st.subheader("Classify a sentence as grammatically correct or incorrect.")

selected_text = st.selectbox("Select a sample Tweet", examples)
custom_input = st.text_input("Try it for yourself!")

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

