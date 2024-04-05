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
    document_assembler = DocumentAssembler()
    document_assembler.setInputCol('text')
    document_assembler.setOutputCol('document')

    # Separates the text into individual tokens (words and punctuation).
    tokenizer = Tokenizer()
    tokenizer.setInputCols(['document'])
    tokenizer.setOutputCol('token')

    # Encodes the text as a single vector representing semantic features.
    sentence_encoder = UniversalSentenceEncoder.pretrained(model)
    sentence_encoder.setInputCols(['document'])
    sentence_encoder.setOutputCol('sentence_embeddings')

    nlp_pipeline = Pipeline(stages=[
        document_assembler, 
        tokenizer,
        sentence_encoder
    ])

    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = nlp_pipeline.fit(empty_df)
    light_pipeline = LightPipeline(pipeline_model)
    return light_pipeline

# def fit_data(pipeline, data):
#     empty_df = spark.createDataFrame([['']]).toDF('text')
#     pipeline_model = pipeline.fit(empty_df)
#     model = LightPipeline(pipeline_model)
#     results = model.fullAnnotate(data)[0]

#     return results['sentence_embeddings'][0].result

def get_similarity(light_pipeline, input_list):
    df = spark.createDataFrame(pd.DataFrame({'text': input_list}))
    result = light_pipeline.transform(df)
    embeddings = []
    for r in result.collect():
        embeddings.append(r.sentence_embeddings[0].embeddings)
    embeddings_matrix = np.array(embeddings)
    return np.matmul(embeddings_matrix, embeddings_matrix.transpose())

############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Detect similar sentences", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.title("Detect similar sentences")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["tfhub_use" , "tfhub_use_lg"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTENCE_SIMILARITY.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

sentences = [
    "Sign up for our mailing list to get free offers and updates about our products!",
    "Subscribe to notifications to receive information about discounts and new offerings.",
    "Send in your information for a chance to win big in our Summer Sweepstakes!",
    "After filling out this form, you will receive a confirmation email to complete your signup.",
    "It was raining, so I waited beneath the balcony outside the cafe.",
    "I stayed under the deck of the cafe because it was rainy outside.",
    "I like the cafe down the street because it's not too loud in there.",
    "The coffee shop near where I live is quiet, so I like to go there.",
    "Web traffic analysis shows that most Internet users browse on mobile nowadays.",
    "The analytics show that modern web users mostly use their phone instead of their computers."
]

st.subheader("Automatically compute the similarity between two sentences using Spark NLP Universal Sentence Embeddings or T5 Transformer.")


col1, col2 = st.columns(2)
with col1:
   sentence1 = st.selectbox('First sentence to compare', sentences)
with col2:
   sentence2 = st.selectbox('Second sentence to compare', sentences)

st.subheader("Try it for yourself!")
col1, col2 = st.columns(2)
with col1:
   custom_sentence1 = st.text_input('First sentence')
with col2:
   custom_sentence2 = st.text_input('Second sentence')

if custom_sentence1 and custom_sentence2:
    selected_list = [custom_sentence1, custom_sentence2]
elif sentence1 and sentence2:
    selected_list = [sentence1, sentence2]

st.subheader('Selected Text')

st.write(f"1: {selected_list[0]}")
st.write(f"2: {selected_list[1]}")

Pipeline = create_pipeline(model)
output = get_similarity(Pipeline, selected_list)

st.subheader('Prediction')

num1 = output[0][0]
num2 = output[0][1]

similarity = round((min(num1, num2) / max(num1, num2)) * 100)

if similarity == 100:
    similarity_str = "identical"
elif similarity >= 80:
    similarity_str = "very similar"
elif similarity >= 60:
    similarity_str = "similar"
elif similarity >= 40:
    similarity_str = "somewhat similar"
else:
    similarity_str = "not similar"
st.markdown(f'Detected similarity: **{similarity}%**')
st.markdown(f'These sentences are **{similarity_str}**.')