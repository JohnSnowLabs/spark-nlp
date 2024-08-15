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

    sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "es") \
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
    layout="wide", page_title="Analyze sentiment in Spanish texts", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Danish Sentiment Analysis")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["bert_sequence_classifier_sentiment"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/BertForSequenceClassification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Hr.formand, kære kommissær og kære kolleger, jeg vil starte med at sige Hr.Rapkay tak for en god betænkning og et godt samarbejde.",
    "Protester over hele landet ledet af utilfredse civilsamfund på grund af den danske regerings COVID-19 lockdown-politik er kommet ud af kontrol.",
    "Hvidbogen repræsenterer tre til fire måneders intensivt arbejde, siden den nye kommission blev udpeget i september.",
    "Det er ikke et nemt emne, og jeg mener derfor, at den indsats, som fru Lienemanns har ydet, fortjener stor respekt.",
    "Det er desværre en sørgelig erkendelse af de store vanskeligheder, som Wales oplever.",
    "Vi har indarbejdet resultaterne af de omfattende høringer, der har fundet sted i løbet af det sidste par år siden offentliggørelsen af kommissionens grønbog om fødevarelovgivning.",
    "Fru formand, jeg vil gerne først give Dem en kompliment for den kendsgerning, at De har holdt Deres ord, og at antallet af tv-kanaler på vores kontorer faktisk er udvidet enormt nu i denne første mødeperiode i det nye år.",
    "Det er fuldstændig skandaløst, at vi fastsætter lovgivningsbestemmelser og så ikke overholder dem selv.",
    "Min gruppe har udarbejdet omfattende ændringsforslag til begge betænkninger til forhandling i dag."
        ]

st.subheader("This model identifies positive, negative or neutral sentiments in Danish texts.")

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
