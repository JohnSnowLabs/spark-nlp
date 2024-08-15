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
def create_pipeline():
    
    document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

    tokenizer = RecursiveTokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")\
    .setPrefixes(["\"", "(", "[", "\n"])\
    .setSuffixes([".", ",", "?", ")","!", "‘s"])

    spell_model = ContextSpellCheckerModel\
        .pretrained('spellcheck_dl')\
        .setInputCols("token")\
        .setOutputCol("corrected")

    light_pipeline = Pipeline(stages = [document_assembler, tokenizer, spell_model])

    return light_pipeline

def fit_data(pipeline, data):
  empty_df = spark.createDataFrame([['']]).toDF('text')
  pipeline_model = pipeline.fit(empty_df)
  model = LightPipeline(pipeline_model)
  result = model.annotate(data)

  return result

        
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Spell check your text documents", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Spell check your text documents")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)
model_list = ["spellcheck_dl"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link = """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Apollo 11 was the space fleght that landed the frrst humans, Americans Neil Armstrong and Buzz Aldrin, on the Mon on july 20, 1969, at 20:18 UTC. Armstrong beceme the first to stp onto the luner surface 6 hours later on July 21 at 02:56 UTC. Armstrong spent abut three and a half two and a hakf hours outside the spacecraft, Aldrin slghtely less; and tgether they colected 47.5 pounds (21.5 kg) of lunar material for returne to Earth. A third member of the mission, Michael Collins, pilloted the comand spacecraft alone in lunar orbit untl Armstrong and Aldrin returned to it for the trep back to Earth.",
    "Set theory is a branch of mathematical logic that studyes sets, which informally are colections of objects. Although any type of object can be collected into a set, set theory is applyed most often to objects that are relevant to mathematics. The language of set theory can be used to define nearly all mathematical objects. Set theory is commonly employed as a foundational system for mathematics. Beyond its foundational role, set theory is a branch of mathematics in its own right, with an active resurch community. Contemporary resurch into set theory includes a divers colection of topics, ranging from the structer of the real number line to the study of the consistency of large cardinals.",
    "In mathematics and transporation enginering, traffic flow is the study of interctions between travellers (including podestrians, ciclists, drivers, and their vehicles) and inferstructure (including highways, signage, and traffic control devices), with the aim of understanding and developing an optimal transport network with eficiant movement of traffic and minimal traffic congestion problems. Current traffic models use a mixture of emperical and theoretical techniques. These models are then developed into traffic forecasts, and take account of proposed local or major changes, such as incrased vehicle use, changes in land use or changes in mode of transport (with people moving from bus to train or car, for example), and to identify areas of congestion where the network needs to be ajusted.",
    "Critical theory is a social pholosiphy pertaning to the reflective asessment and critique of society and culture in order to reveal and challange power structures. With origins in socology, as well as in literary criticism, it argues that social problems are influenced and created more by societal structures and cultural assumptions than by individual and psychological factors. Mantaining that ideology is the principal obsticle to human liberation, critical theory was establiched as a school of thought primarily by the Frankfurt School theoreticians. One sociologist described a theory as critical insofar as it seeks 'to liberate human beings from the circomstances that enslave them.",
    "In computitional languistics, lemmatisation is the algorhythmic process of determining the lemma of a word based on its intended meaning. Unlike stemming, lemmatisation depends on corectly identifing the intended part of speech and meaning of a word in a sentence, as well as within the larger context surounding that sentence, such as neigboring sentences or even an entire document. As a result, devleoping efficient lemmatisation algorithums is an open area of research. In many langauges, words appear in several inflected forms. For example, in English, the verb 'to walk' may appear as 'walk', 'walked', 'walks' or 'walking'. The base form, 'walk', that one might look up in a dictionery, is called the lemma for the word. The asociation of the base form with a part of speech is often called a lexeme of the word."
]

st.subheader("Spark NLP contextual spellchecker allows the quick identification of typos or spelling issues within any text document.")

selected_text = st.selectbox("Select an example", examples)
custom_input = st.text_input("Try it for yourself!")

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader("Selected Text:")
st.write(selected_text)

st.subheader(f"Misspells Detected:")
Pipeline = create_pipeline()
output = fit_data(Pipeline, selected_text)

correction_dict_filtered = {token: corrected for token, corrected in zip(output['token'], output['corrected']) if token != corrected}

def generate_html_with_corrections(sentence, corrections):
    corrected_html = sentence
    for incorrect, correct in corrections.items():
        corrected_html = corrected_html.replace(incorrect, f'<span style="text-decoration: line-through; color: red;">{incorrect}</span> <span style="color: green;">{correct}</span>')
    return f'<div style="font-family: Arial, sans-serif; font-size: 16px;">{corrected_html}</div>'

corrected_html_snippet = generate_html_with_corrections(selected_text, correction_dict_filtered)
st.markdown(corrected_html_snippet, unsafe_allow_html=True)





