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
      .pretrained('sent_bert_wiki_books_sst2', 'en') \
      .setInputCols(["document"])\
      .setOutputCol("sentence_embeddings")

  sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bertwiki_finance_sentiment", "en") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class_")

  financial_sentiment_pipeline = Pipeline(
      stages=[document, 
              embeddings, 
              sentimentClassifier])
  
  return financial_sentiment_pipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['class_'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Sentiment Analysis of Financial news", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Sentiment Analysis of Financial news")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_bertwiki_finance_sentiment"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_FINANCE.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
  "In April 2005 , Neste separated from its parent company , Finnish energy company Fortum , and became listed on the Helsinki Stock Exchange .",
  "Finnish IT solutions provider Affecto Oyj HEL : AFE1V said today its slipped to a net loss of EUR 115,000 USD 152,000 in the second quarter of 2010 from a profit of EUR 845,000 in the corresponding period a year earlier .",
  "10 February 2011 - Finnish media company Sanoma Oyj HEL : SAA1V said yesterday its 2010 net profit almost tripled to EUR297 .3 m from EUR107 .1 m for 2009 and announced a proposal for a raised payout .",
  "Profit before taxes decreased by 9 % to EUR 187.8 mn in the first nine months of 2008 , compared to EUR 207.1 mn a year earlier .",
  "The world 's second largest stainless steel maker said net profit in the three-month period until Dec. 31 surged to euro603 million US$ 781 million , or euro3 .33 US$ 4.31 per share , from euro172 million , or euro0 .94 per share , the previous year .",
  "TietoEnator signed an agreement to acquire Indian research and development ( R&D ) services provider and turnkey software solutions developer Fortuna Technologies Pvt. Ltd. for 21 mln euro ( $ 30.3 mln ) in September 2007 ."
]

st.subheader("This model identifies positive, negative or neutral sentiments in financial news.")

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
  st.markdown("""<h3>This seems like <span style="color: green">{}</span> news. <span style="font-size:35px;">&#128515;</span></h3>""".format('positive'), unsafe_allow_html=True)
elif output.lower() in ['neg', 'negative']:
  st.markdown("""<h3>This seems like <span style="color: red">{}</span> news. <span style="font-size:35px;">&#128544;</span?</h3>""".format('negative'), unsafe_allow_html=True)
elif output in ['neutral', 'normal']:
  st.markdown("""<h3>This seems like <span style="color: #209DDC">{}</span> news. <span style="font-size:35px;">&#128578;</span></h3>""".format(output), unsafe_allow_html=True)
