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
  document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  sentence_detector = SentenceDetector() \
      .setInputCols(["document"]) \
      .setOutputCol("sentence")

  tokenizer = Tokenizer() \
      .setInputCols(["sentence"]) \
      .setOutputCol("token")

  word_embeddings = WordEmbeddingsModel()\
      .pretrained('urduvec_140M_300d', 'ur')\
      .setInputCols(["sentence",'token'])\
      .setOutputCol("word_embeddings")

  sentence_embeddings = SentenceEmbeddings() \
      .setInputCols(["sentence", "word_embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

  classifier = SentimentDLModel.pretrained('sentimentdl_urduvec_imdb', 'ur' )\
      .setInputCols(['sentence_embeddings'])\
      .setOutputCol('sentiment')

  nlpPipeline = Pipeline(
      stages=[ 
          document_assembler,
          sentence_detector, 
          tokenizer,
          word_embeddings,
          sentence_embeddings,
          classifier ])
  
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
    layout="wide", page_title="Analyze sentiment in Urdu", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Analyze sentiment in Urdu movie reviews and tweets")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["sentimentdl_urduvec_imdb"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/SENTIMENT_UR.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

folder_path = f"/content/Sentiment UR/inputs/{model}"
examples = [lines[1].strip() for filename in os.listdir(folder_path) if filename.endswith('.txt') for lines in [open(os.path.join(folder_path, filename), 'r').readlines()] if len(lines) >= 2]

st.subheader("Detect the general sentiment expressed in a movie review or tweet by using our pretrained Spark NLP sentiment analysis model.")

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

if output.lower() in ['pos', 'positive']:
  st.markdown("""<h3>This seems like a <span style="color: green">{}</span> text. <span style="font-size:35px;">&#128515;</span></h3>""".format('positive'), unsafe_allow_html=True)
elif outputlower() in ['neg', 'negative']:
  st.markdown("""<h3>This seems like a <span style="color: red">{}</span> text. <span style="font-size:35px;">&#128544;</span?</h3>""".format('negative'), unsafe_allow_html=True)
