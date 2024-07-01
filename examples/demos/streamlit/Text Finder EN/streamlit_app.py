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
def create_pipeline(model, matches):
    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
    if model == 'TextMatcher':
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        text_matcher = TextMatcher() \
            .setInputCols(["document",'token'])\
            .setOutputCol("matched_text")\
            .setCaseSensitive(False)\
            .setEntities(path=matches)
        nlpPipeline = Pipeline(stages=[documentAssembler, tokenizer, text_matcher])

    else:
        regex_matcher = RegexMatcher()\
            .setInputCols('document')\
            .setStrategy("MATCH_ALL")\
            .setOutputCol("matched_text")\
            .setExternalRules(path=matches, delimiter=',')
        nlpPipeline = Pipeline(stages=[documentAssembler, regex_matcher])

    return nlpPipeline

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    res_list = []
    for res in results['matched_text']:
        res_list.append(res.result)

    return res_list

def highlight_words(word_list, sentence):
    import html
    sentence = html.escape(sentence)
    for word in word_list:
        sentence = sentence.replace(word, f'<span style="background-color: pink;">{word}</span>')
    return sentence
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Text Finder", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")
st.title("Find text in a document")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
st.sidebar.image(logo_path, width=300)

model_list = ["TextMatcher", "RegexMatcher"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# Reference notebook link in sidebar
link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TEXT_FINDER_EN.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

folder_path = f"/content/Text Finder EN/inputs/{model}"
examples = [lines[1].strip() for filename in os.listdir(folder_path) if filename.endswith('.txt') for lines in [open(os.path.join(folder_path, filename), 'r').readlines()] if len(lines) >= 2]

st.subheader("Finds a text in document either by keyword or by regex expression.")
selected_text = st.selectbox("Select an example", examples)
custom_input = st.text_input("Try it with your own Sentence!")

text_to_analyze = custom_input if custom_input else selected_text

st.subheader('Text to analyze')
st.write(text_to_analyze)

if model == 'RegexMatcher':
    custom_input = st.subheader("List of Regular Expressions to match (case sensitive):")
    txt_path = "/content/Text Finder EN/inputs/RegexMatcher/regex_to_match.txt"
else:
    custom_input = st.subheader("Try it for yourself! Enter the Words you want to search for")
    txt_path = "/content/Text Finder EN/inputs/TextMatcher/text_to_match.txt"

conditions = open(os.path.join(txt_path), 'r').readlines()
text_wrapper = """<div class="scroll entities" style="overflow-x: auto; height: 200px; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem; white-space:pre-wrap">{}</div>"""
st.markdown(text_wrapper.format("".join(conditions)), unsafe_allow_html=True)

custom_labels = st.text_area("Try it with your own Labels!")
custom_file_path = '/content/Text Finder EN/inputs/custom_matches.txt'

if len(custom_labels) > 0:
    with open(custom_file_path, 'a') as file:
        file.write(custom_labels)
else:
    with open(custom_file_path, 'w') as file:
        file.truncate()

labels_to_match = custom_file_path if custom_labels else txt_path

pipeline = create_pipeline(model, labels_to_match)
output = fit_data(pipeline, text_to_analyze)

st.subheader("Matched sentence:")

highlighted_html = highlight_words(output, text_to_analyze)
st.markdown(highlighted_html, unsafe_allow_html=True)
