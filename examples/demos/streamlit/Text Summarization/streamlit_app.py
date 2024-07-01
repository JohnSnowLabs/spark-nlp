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
    document_assembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("documents")

    t5 = T5Transformer() \
        .pretrained(model, 'en') \
        .setTask("summarize:")\
        .setMaxOutputLength(200)\
        .setInputCols(["documents"]) \
        .setOutputCol("summaries")
    
    pipeline = Pipeline().setStages([document_assembler, t5])

    return pipeline

def fit_data(pipeline, data):
    #df = spark.createDataFrame([[data]]).toDF('text')
    #results = pipeline.fit(df).transform(df)

    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['summaries'][0].result
    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Text Summarization", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Summarize Text")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)
model = st.sidebar.selectbox("Choose the pretrained model", ['t5_base', 't5_small'], help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

options = [

    "Mount Tai is a mountain of historical and cultural significance located north of the city of Tai'an, in Shandong province, China. The tallest peak is the Jade Emperor Peak, which is commonly reported as being 1,545 meters tall, but is officially described by the PRC government as 1,532.7 meters tall. It is associated with sunrise, birth, and renewal, and is often regarded the foremost of the five. Mount Tai has been a place of worship for at least 3,000 years and served as one of the most important ceremonial centers of China during large portions of this period.",
    "The Guadeloupe amazon (Amazona violacea) is a hypothetical extinct species of parrot that is thought to have been endemic to the Lesser Antillean island region of Guadeloupe. Described by 17th- and 18th-century writers, it is thought to have been related to, or possibly the same as, the extant imperial amazon. A tibiotarsus and an ulna bone from the island of Marie-Galante may belong to the Guadeloupe amazon. According to contemporary descriptions, its head, neck and underparts were mainly violet or slate, mixed with green and black; the back was brownish green; and the wings were green, yellow and red. It had iridescent feathers, and was able to raise a \"ruff\" of feathers around its neck. It fed on fruits and nuts, and the male and female took turns sitting on the nest. French settlers ate the birds and destroyed their habitat. Rare by 1779, the species appears to have become extinct by the end of the 18th century.",
    "Pierre-Simon, marquis de Laplace (23 March 1749 – 5 March 1827) was a French scholar and polymath whose work was important to the development of engineering, mathematics, statistics, physics, astronomy, and philosophy. He summarized and extended the work of his predecessors in his five-volume Mécanique Céleste (Celestial Mechanics) (1799–1825). This work translated the geometric study of classical mechanics to one based on calculus, opening up a broader range of problems. In statistics, the Bayesian interpretation of probability was developed mainly by Laplace.",
    "John Snow (15 March 1813 – 16 June 1858) was an English physician and a leader in the development of anaesthesia and medical hygiene. He is considered one of the founders of modern epidemiology, in part because of his work in tracing the source of a cholera outbreak in Soho, London, in 1854, which he curtailed by removing the handle of a water pump. Snow's findings inspired the adoption of anaesthesia as well as fundamental changes in the water and waste systems of London, which led to similar changes in other cities, and a significant improvement in general public health around the world.",
    "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as \"the best known, the most visited, the most written about, the most sung about, the most parodied work of art in the world\". The painting's novel qualities include the subject's enigmatic expression, the monumentality of the composition, the subtle modelling of forms, and the atmospheric illusionism.",
    """Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations. It has two major branches, differential calculus and integral calculus; the former concerns instantaneous rates of change, and the slopes of curves, while integral calculus concerns accumulation of quantities, and areas under or between curves. These two branches are related to each other by the fundamental theorem of calculus, and they make use of the fundamental notions of convergence of infinite sequences and infinite series to a well-defined limit.[1] Infinitesimal calculus was developed independently in the late 17th century by Isaac Newton and Gottfried Wilhelm Leibniz.[2][3] Today, calculus has widespread uses in science, engineering, and economics.[4] In mathematics education, calculus denotes courses of elementary mathematical analysis, which are mainly devoted to the study of functions and limits. The word calculus (plural calculi) is a Latin word, meaning originally "small pebble" (this meaning is kept in medicine – see Calculus (medicine)). Because such pebbles were used for calculation, the meaning of the word has evolved and today usually means a method of computation. It is therefore used for naming specific methods of calculation and related theories, such as propositional calculus, Ricci calculus, calculus of variations, lambda calculus, and process calculus.""",
]

st.subheader("Summarize text to make it shorter while retaining meaning.")

selected_text = st.selectbox("Select an example", options)
custom_input = st.text_input("Try it for yourself!")

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader('Text')
st.write(selected_text)

st.subheader("Summary")

Pipeline = create_pipeline(model)
output = fit_data(Pipeline, selected_text)
st.write(output)
