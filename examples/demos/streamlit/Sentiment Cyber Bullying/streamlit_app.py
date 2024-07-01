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
    layout="wide", page_title="Detect cyberbullying in tweets", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Detect cyberbullying in tweets")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_use_cyberbullying"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_CYBERBULLYING.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
  "@CALMicC he kept me informed on stuff id missed and seemed ok. I liked him.",
  "@AMohedin Okay, we have women being physically inferior and the either emotionally or mentally inferior in some way.",
  "@LynnMagic people think that implying association via follow is a bad thing. but it's shockingly accurate.",
  "@Rayandawlah_ @_Jihad10 These days might and honor come from science, technology, humanitarianism. Which is why Muslims won't get any.",
  "Stay outve Congress and we have a deal. @jacobkramer17 Call me sexist bt the super bowl should b guys only no women are allowed n th stadium",
  "I'm looking for a few people to help with @ggautoblocker's twitter. Log &amp; categorize mentions as support requests/abusive/positive tweets.",
  "@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked",
  """@ListenToRaisin No question. Feminists have the media. Did you see any mention of Clem Fords OPEN bigotry, etc?  Nope. "Narrative" is all.""",
  "RT @EBeisner @ahall012 I agree with you!! I would rather brush my teeth with sandpaper then watch football with a girl!!",
  "@hibach8 But it is a lie.  The religion is a disgusting, terrorist, hate mongering piece of filth.  That has nothing to do with individuals."
]

st.subheader("Identify Racism, Sexism or Neutral tweets using our pretrained emotions detector.")

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

if output.lower() in ['neutral', 'normal']:
  st.markdown("""<h3>This seems like a <span style="color: green">{}</span> tweet. <span style="font-size:35px;">&#128515;</span></h3>""".format(output), unsafe_allow_html=True)
elif output.lower() in ['racism', 'sexism']:
  st.markdown("""<h3>This seems like a <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#129324;</span></h3>""".format(output), unsafe_allow_html=True)