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
    document = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    embeddings = BertSentenceEmbeddings\
        .pretrained('labse', 'xx') \
        .setInputCols(["document"])\
        .setOutputCol("sentence_embeddings")

    sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "fr") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class_")    
        
    nlpPipeline = Pipeline(
        stages=[
            document, 
            embeddings,
            sentimentClassifier])

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
    layout="wide", page_title="Sentiment Analysis of French texts", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Sentiment Analysis of French texts")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_bert_sentiment"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_Fr_Sentiment.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Jeu et championnat pas assez excitants ? Devez-vous vérifier l'arbitre vidéo maintenant? Je suis horrifié, pensa-t-il, il ne devrait répondre que s'il a fait des erreurs flagrantes. Le football n'est plus amusant.",
    "J'ai raté le podcast werder hier mercredi. À quelle vitesse vous vous habituez à quelque chose et vous l'attendez avec impatience. Merci à Plainsman pour les bonnes interviews et la perspicacité dans les coulisses du werderbremen. Passez de bonnes vacances d'hiver !",
    "Les scènes s'enchaînent de manière saccadée, les dialogues sont théâtraux, le jeu des acteurs ne transcende pas franchement le film. Seule la musique de Vivaldi sauve le tout. Belle déception.",
    "Je n'aime pas l'arbitre parce qu'il est empoisonné !",
    "ManCity Guardiola et sa bande, vous êtes des connards. Je viens de perdre une fortune à cause de ta dette envers tes Bavarois là-bas."
    ]
st.subheader("This model identifies the sentiments (positive or negative) in French texts.")

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
