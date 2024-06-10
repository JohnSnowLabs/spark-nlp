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
  documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
  use = UniversalSentenceEncoder.pretrained("tfhub_use", "en")\
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
    layout="wide", page_title="Detect emotions in tweets", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Detect emotions in tweets")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_use_emotion"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_EMOTION.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
  "I am SO happy the news came out in time for my birthday this weekend! My inner 7-year-old cannot WAIT!",
  "That moment when you see your friend in a commercial. Hahahaha!",
  "My soul has just been pierced by the most evil look from @rickosborneorg. A mini panic attack &amp; chill in bones followed soon after.",
  "For some reason I woke up thinkin it was Friday then I got to school and realized its really Monday -_-",
  "I'd probably explode into a jillion pieces from the inablility to contain all of my if I had a Whataburger patty melt right now. #drool",
  "These are not emotions. They are simply irrational thoughts feeding off of an emotion",
  "Found out im gonna be with sarah bo barah in ny for one day!!! Eggcitement :)",
  "That awkward moment when you find a perfume box full of sensors!",
  "Just home from group celebration - dinner at Trattoria Gianni, then Hershey Felder's performance - AMAZING!!",
  "Nooooo! My dad turned off the internet so I can't listen to band music!"
  ]

st.subheader("Automatically identify Joy, Surprise, Fear, Sadness in Tweets using out pretrained Spark NLP DL classifier.")

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

if output == 'joy':
  st.markdown("""<h3>This seems like a <span style="color: #f0a412">{}</span> tweet. <span style="font-size:35px;">&#128514;</span></h3>""".format('joyous'), unsafe_allow_html=True)
elif output == 'surprise':
  st.markdown("""<h3>This seems like a <span style="color: #209DDC">{}</span> tweet. <span style="font-size:35px;">&#128522;</span></h3>""".format('surprised'), unsafe_allow_html=True)
elif output == 'sadness':
  st.markdown("""<h3>This seems like a <span style="color: #8F7F6C">{}</span> tweet. <span style="font-size:35px;">&#128543;</span></h3>""".format('sad'), unsafe_allow_html=True)
elif output == 'fear':
  st.markdown("""<h3>This seems like a <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#128561;</span></h3>""".format('fearful'), unsafe_allow_html=True)
