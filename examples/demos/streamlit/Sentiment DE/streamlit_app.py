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
    document = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    embeddings = BertSentenceEmbeddings\
        .pretrained('labse', 'xx') \
        .setInputCols(["document"])\
        .setOutputCol("sentence_embeddings")

    sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "de") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class_")   
        
    nlpPipeline = Pipeline(
        stages=[
        document, 
        embeddings,
        sentimentClassifier
        ])
    
    return nlpPipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['class_'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Sentiment Analysis of German texts", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Sentiment Analysis of German texts")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_bert_sentiment"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_De_SENTIMENT.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Spiel und Meisterschaft nicht spannend genug? Muss man jetzt den Videoschiedsrichter kontrollieren? Ich bin entsetzt,dachte der darf nur bei krassen Fehlentscheidungen ran. So macht der Fussball keinen Spass mehr.",
    "Habe gestern am Mittwoch den werder Podcast vermisst. Wie schnell man sich an etwas gewöhnt und darauf freut. Danke an Plainsman für die guten Interviews und den Einblick hinter die Kulissen von werderbremen. Angenehme Winterpause weiterhin!",
    "Die Szenen folgen ruckartig aufeinander, die Dialoge sind theatralisch, die schauspielerischen Leistungen kommen nicht wirklich über den Film hinaus. Nur die Musik von Vivaldi rettet den Tag. Eine große Enttäuschung.",
    "ich mag den Schiri nicht, denn der ist vergiftet!",
    "ManCity Guardiola und seine Gang, ihr seid Arschlöcher. Ich habe gerade ein Vermögen durch deine Schuld deine Bayern dort verloren"
    ]

st.subheader("This model identifies the sentiments (positive or negative) in German texts.")

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
