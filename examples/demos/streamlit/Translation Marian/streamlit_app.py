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

def create_pipeline(model):
    documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

    ## More accurate Sentence Detection using Deep Learning
    sentencerDL = SentenceDetectorDLModel()\
      .pretrained("sentence_detector_dl", "xx")\
      .setInputCols(["document"])\
      .setOutputCol("sentences")

    marian = MarianTransformer.pretrained(model, "xx")\
      .setInputCols(["sentences"])\
      .setOutputCol("translation")

    nlp_pipeline = Pipeline(
        stages=[
            documentAssembler, 
            sentencerDL, 
            marian
            ])

    return nlp_pipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['translation'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Translate text", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Translate text")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
st.sidebar.image(logo_path, width=300)


model_mappings = {
    "opus_mt_en_fr": "Translate text from English to French",
    "opus_mt_en_it": "Translate text from English to Italian",
    "opus_mt_en_es": "Translate text from English to Spanish",
    "opus_mt_en_de": "Translate text from English to German",
    "opus_mt_en_cpp": "Translate text from English to Portuguese",
    "opus_mt_fr_en": "Translate text from French to English",
    "opus_mt_it_en": "Translate text from Italian to English",
    "opus_mt_es_en": "Translate text from Spanish to English",
    "opus_mt_de_en": "Translate text from German to English",
    "opus_mt_cpp_en": "Translate text from Portuguese to English"
}

st.sidebar.title("Select Languages:")

lang_map_dict={"English": 'en', 'French': 'fr','Italian': 'it', 'Spanish': 'es', 'German': 'de', 'Portuguese': 'cpp'}
from_lang_map_dict={"English": 'en','French': 'fr','Italian': 'it', 'Spanish': 'es', 'German': 'de', 'Portuguese': 'cpp'}
lang_names = list(from_lang_map_dict.keys())
from_lang = st.sidebar.selectbox("From", lang_names)

if from_lang == 'English':
    to_lang = st.sidebar.selectbox("To", ['French', 'Italian', 'Spanish', 'German', 'Portuguese'])
else:
    to_lang = st.sidebar.selectbox("To", ['English'])

selected_model = f'opus_mt_{lang_map_dict[from_lang]}_{lang_map_dict[to_lang]}'
st.title(model_mappings[selected_model])

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TRANSLATION_MARIAN.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

folder_path = f"/content/Translation Marian/inputs/{selected_model}"
examples = [lines[1].strip() for filename in os.listdir(folder_path) if filename.endswith('.txt') for lines in [open(os.path.join(folder_path, filename), 'r').readlines()] if len(lines) >= 2]

selected_text = st.selectbox("Select a sample text", examples)
custom_input = st.text_input("Try it for yourself!")

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader('Full example text')
st.write(selected_text)

st.subheader("Translation")

Pipeline = create_pipeline(selected_model)
output = fit_data(Pipeline, selected_text)
st.write(output)