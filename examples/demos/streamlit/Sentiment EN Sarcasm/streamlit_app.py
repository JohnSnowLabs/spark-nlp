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

def create_pipeline(model):
  documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")
    
  use = UniversalSentenceEncoder.pretrained()\
  .setInputCols(["document"])\
  .setOutputCol("sentence_embeddings")

  sentimentdl = ClassifierDLModel.pretrained(model)\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

  nlpPipeline = Pipeline(stages = [documentAssembler, use, sentimentdl])
  
  return nlpPipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['sentiment'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Detect sarcastic tweets", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Detect sarcastic tweets")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_use_sarcasm"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_SARCASM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
  "Love getting home from work knowing that in less than 8hours you're getting up to go back there again.",
  "Oh my gosh! Can you imagine @JessieJ playing piano on her tour while singing a song. I would die and go to heaven. #sheisanangel",
  "Dear Teva, thank you for waking me up every few hours by howling. Your just trying to be mother natures alarm clock.",
  "The United States is a signatory to this international convention",
  "If I could put into words how much I love waking up at am on Tuesdays I would",
  "@pdomo Don't forget that Nick Foles is also the new Tom Brady. What a preseason! #toomanystudQBs #thankgodwedonthavetebow",
  "I cant even describe how excited I am to go cook noodles for hours",
  "@Will_Piper should move back up fella. I'm already here... On my own... Having loads of fun",
  "Tweeting at work... Having sooooo much fun and honestly not bored at all #countdowntillfinish",
  "I can do what I want to. I play by my own rules"
]

st.subheader("Checkout our sarcasm detection pretrained Spark NLP model. It is able to tell apart normal content from sarcastic content.")

selected_text = st.selectbox("Select a sample", examples)
custom_input = st.text_input("Try it for yourself!")

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader('Selected Text')
st.write(selected_text)

Pipeline = create_pipeline(model)
output = fit_data(Pipeline, selected_text)

if output in ['neutral', 'normal']:
  st.markdown("""<h3>This seems like <span style="color: #209DDC">{}</span> news. <span style="font-size:35px;">&#128578;</span></h3>""".format(output), unsafe_allow_html=True)
elif output == 'sarcasm':
  st.markdown("""<h3>This seems like a <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#128579;</span></h3>""".format('sarcastic'), unsafe_allow_html=True)