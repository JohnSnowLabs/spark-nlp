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

def create_pipeline(model, task):
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("documents")
        
    t5 = T5Transformer.pretrained(model) \
        .setTask(task) \
        .setInputCols(["documents"])\
        .setMaxOutputLength(200)\
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
    layout="wide", page_title="T5 Transformers", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
st.sidebar.image(logo_path, width=300)

model_list = {
    "t5_informal_to_formal_styletransfer": "T5 for Informal to Formal Style Transfer",
    "t5_formal_to_informal_styletransfer": "T5 for Formal to Informal Style Transfer"
}

model = st.sidebar.selectbox("Choose the pretrained model", model_list.keys(), help="For more info about the models visit: https://sparknlp.org/models",)
st.title(model_list[model])

# Reference notebook link in sidebar
link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

examples = {
    "t5_informal_to_formal_styletransfer": [
        "Who gives a crap?",
        "Hiya, how ya doing?",
        "btw - ur looks familiar.",
        "I looooooooooooooove going to the movies."
    ],
    "t5_formal_to_informal_styletransfer": [
        "Please leave the room now.",
        "Thank you, sir!",
        "It's a pleasure to meet you.",
        "I appreciate your assistance.",
        "She understood very quickly.",
        "He contracted a fever.",
        "He investigated his accountant."
    ]
}

task_descriptions = {
    "t5_informal_to_formal_styletransfer": "Transfer Casual to Formal:",
    "t5_formal_to_informal_styletransfer": "Transfer Formal to Casual:"
}

st.subheader("This is a text-to-text T5 model which is fine-tuned for style transfer.")
selected_text = st.selectbox("Select an example", examples[model])
custom_input = st.text_input("Try it for yourself!")

text_to_analyze = custom_input if custom_input else selected_text

st.subheader('Text to analyze')
st.write(text_to_analyze)

st.subheader("Predicted sentence:")
pipeline = create_pipeline(model, task_descriptions[model])
output = fit_data(pipeline, text_to_analyze)
st.write(output)
