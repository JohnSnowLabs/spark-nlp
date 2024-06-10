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
    documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

    sentencerDL = SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentences")

    sd_pipeline = PipelineModel(stages=[documenter, sentencerDL])

    return sd_pipeline

def fit_data(pipeline, data):
    model = LightPipeline(pipeline)
    results = model.fullAnnotate(data)[0]
    return results
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Detect sentences in text", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.title("Detect sentences in text")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["sentence_detector_dl"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Brexit: U.K to ban more EU citizens with criminal records. In a recent Home Office meeting a consensus has been made to tighten border policies. However a no-deal Brexit could make it harder to identify foreign criminals; With the U.K in a transition period since it formally left the EU in January, an EU citizen can currently only be refused entry if they present a genuine, present and serious threat.",
    """Harry Harding on the US, China, and a ‘Cold War 2.0’. “Calling it a second Cold War is misleading, but to deny that it’s a Cold War is also disingenuous.”, Harding is a specialist on Asia and U.S.-Asian relations. His major publications include Organizing China: The Problem of Bureaucracy, 1949-1966.The phrase “new Cold War” is an example of the use of analogies in understanding the world.The world is a very complicated place.People like to find ways of coming to a clearer and simpler understanding.""",
    "Tesla’s latest quarterly numbers beat analyst expectations on both revenue and earnings per share, bringing in $8.77 billion in revenues for the third quarter.That’s up 39% from the year-ago period.Wall Street had expected $8.36 billion in revenue for the quarter, according to estimates published by CNBC. Revenue grew 30% year-on-year, something the company attributed to substantial growth in vehicle deliveries, and operating income also grew to $809 million, showing improving operating margins to 9.2%.",
    "2020 is another year that is consistent with a rapidly changing Arctic.Without a systematic reduction in greenhouse gases, the likelihood of our first ‘ice-free’ summer will continue to increase by the mid-21st century;It is already well known that a smaller ice sheet means less of a white area to reflect the sun’s heat back into space. But this is not the only reason the Arctic is warming more than twice as fast as the global average",
    "HBR: The world is changing in rapid, unprecedented ways, but one thing remains certain: as businesses look to embed lessons learned in recent months and to build enterprise resilience for the future, they are due for even more transformation.As such, most organizations are voraciously evaluating existing and future technologies to see if they’ll be able to deliver the innovation at scale that they’ll need to survive and thrive.However, technology should not be central to these transformation efforts; people should."
]

st.subheader("Detect sentences from general purpose text documents using a deep learning model capable of understanding noisy sentence structures.")

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

st.subheader('Detected Sentences')

import pandas as pd

data = [{
    'begin': sentence.begin,
    'end': sentence.end,
    'result': sentence.result
} for sentence in output['sentences']]

df = pd.DataFrame(data)
st.table(df)   
