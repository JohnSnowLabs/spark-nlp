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
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
        
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized")

    stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_sw", "sw") \
            .setInputCols(["normalized"]) \
            .setOutputCol("cleanTokens")\
            .setCaseSensitive(False)

    embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base_finetuned_swahili", "sw")\
        .setInputCols(["document", "cleanTokens"])\
        .setOutputCol("embeddings")

    embeddingsSentence = SentenceEmbeddings() \
        .setInputCols(["document", "embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")

    sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_xlm_roberta_sentiment", "sw") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class_")

    sw_pipeline = Pipeline(
        stages=[
            document_assembler, 
            tokenizer, 
            normalizer, 
            stopwords_cleaner, 
            embeddings, 
            embeddingsSentence, 
            sentimentClassifier
            ])

    return sw_pipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['class_'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Sentiment Analysis of Swahili texts", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Sentiment Analysis of Swahili texts")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_xlm_roberta_sentiment"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_SW.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Tukio bora katika sinema ilikuwa wakati Gerardo anajaribu kupata wimbo ambao unaendelea kupitia kichwa chake.",
    "Ni dharau kwa akili ya mtu na upotezaji mkubwa wa pesa",
    "Kris Kristoffersen ni mzuri kwenye sinema hii na kweli hufanya tofauti.",
    "Hadithi yenyewe ni ya kutabirika tu na ya uvivu.",
    "Ninapendekeza hizi kwa kuwa zinaonekana nzuri sana, kifahari na nzuri",
    "Safaricom si muache kucheza na mkopo wa nambari yangu tafadhali. mnanifilisisha😓😓😯",
    "Bidhaa ilikuwa bora na inafanya kazi vizuri kuliko ya verizon na bei ilikuwa rahisi ",
    "Siwezi kuona jinsi sinema hii inavyoweza kuwa msukumo kwa mtu yeyote kushinda woga na kukataliwa.",
    "Sinema hii inasawazishwa vizuri na vichekesho na mchezo wa kuigiza na nilijifurahisha sana."
]

st.subheader("This model identifies positive or negative sentiments in Swahili texts.")

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
