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
        .setOutputCol("document")

  tokenizer = Tokenizer() \
      .setInputCols("document") \
      .setOutputCol("token")

  sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_toxic","en") \
      .setInputCols(["document", "token"]) \
      .setOutputCol("class")

  pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

  return pipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    st.write(results)
    return results['class'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Detect toxic comments", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Detect toxic comments")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["bert_classifier_toxic"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/BertForSequenceClassification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
  "Streatham again Thank you for the copyedits I have been considering withdrawing the nomination owing to the nonconstructive arguing with a certain reviewer but your edits make me think I cant let everyone down Thank you",
  "Do you know how pathetic you look with your silly little crusade of removing England or English with articles? Grow up and find something better to do with your life.",
  "it is obvius to, me, a black man, that orange is for da black Gs, blue is for mexicans and yellow is 4 whites. coz whites look like woman. and mexicans try to compete with da nigger Gs but we r too strong. but in realty the whites also lose to da black man.",
  "She should stop sticking her tongue out before someone rubs their ass on it. Filthy bitch!!!",
  "personal attacks dont tell me what i can and can not do i'll come over there right now and throw a brick through ur window",
  "First i vandalized his page, then i vandalized again, and then his IP adress was blocked hahahahahahahahahahahahahahahahahahaha. This plan is so evil and yet so much fun.",
  "Silly guy. You should use email like the wikijews",
  """Quick question about the section re-ordering... Why was Antagonists switched with Supporting, and why were all ""bad"" characters removed from the Supporting section?  (converse) " """,
  "I will respond until Armenian scum will stop",
  "Isn't tart a kind of whore or concubine or courtesane Eg she's a politician's tart",
  "....thats why I kill you and your father and your mother and you brother",
  "You're ridiculous It's obvious that you have a problem with people with money and education.  Leave the Horace Mann page alone.  You know nothing of the school whatsoever.  HM doesn't admit dirt."
    
]

st.subheader("Classify comments and tweets into Toxic, Insults, Hate, Obscene, Threat.")

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

if output == 'severe_toxic':
  st.markdown("""<h3>This seems like a <span style="color: #209DDC">{}</span> tweet. <span style="font-size:35px;">&#129324;</span></h3>""".format(out), unsafe_allow_html=True)
elif output == 'toxic':
  st.markdown("""<h3>This seems like a <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#129324;</span></h3>""".format(output), unsafe_allow_html=True)
elif output == 'insult':
  st.markdown("""<h3>This seems like an <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#128560;</span></h3>""".format('insulting'), unsafe_allow_html=True)
elif output == 'identity_hate':
  st.markdown("""<h3>This seems like a <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#128560;</span></h3>""".format(output), unsafe_allow_html=True)
elif output == 'obscene':
  st.markdown("""<h3>This seems like an <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#129324;</span></h3>""".format(output), unsafe_allow_html=True)
elif output == 'threat':
  st.markdown("""<h3>This seems like a <span style="color: #B64434">{}</span> tweet. <span style="font-size:35px;">&#129324;</span></h3>""".format('threatening'), unsafe_allow_html=True)
