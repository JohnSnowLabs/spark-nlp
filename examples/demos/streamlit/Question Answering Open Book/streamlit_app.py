############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os
import pandas as pd
import numpy as np

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
            .pretrained(model) \
            .setTask("question:")\
            .setMaxOutputLength(200)\
            .setInputCols(["documents"]) \
            .setOutputCol("answers")

        t5_pp = Pipeline(stages=[ document_assembler, 
                                    t5])
        return t5_pp

def fit_data(pipeline, data):
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = pipeline.fit(empty_df)
    model = LightPipeline(pipeline_model)
    results = model.fullAnnotate(data)[0]

    return results['answers'][0].result

############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Automatically Answer Questions", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.title("Automatically Answer Questions (Open Book)")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["t5_small"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/QUESTION_ANSWERING_OPEN_BOOK.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = {
    """What does increased oxygen concentrations in the patient’s lungs displace?""": """Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.""",
    """What category of game is Legend of Zelda: Twilight Princess?""": """The Legend of Zelda: Twilight Princess (Japanese: ゼルダの伝説 トワイライトプリンセス, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii version was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in December 2006.""",
    """Who is founder of Alibaba Group?""": """Alibaba Group founder Jack Ma has made his first appearance since Chinese regulators cracked down on his business empire. His absence had fuelled speculation over his whereabouts amid increasing official scrutiny of his businesses. The billionaire met 100 rural teachers in China via a video meeting on Wednesday, according to local government media. Alibaba shares surged 5% on Hong Kong's stock exchange on the news.""",
    """For what instrument did Frédéric write primarily for?""": """Frédéric François Chopin (/ˈʃoʊpæn/; French pronunciation: ​[fʁe.de.ʁik fʁɑ̃.swa ʃɔ.pɛ̃]; 22 February or 1 March 1810 – 17 October 1849), born Fryderyk Franciszek Chopin,[n 1] was a Polish and French (by citizenship and birth of father) composer and a virtuoso pianist of the Romantic era, who wrote primarily for the solo piano. He gained and has maintained renown worldwide as one of the leading musicians of his era, whose "poetic genius was based on a professional technique that was without equal in his generation." Chopin was born in what was then the Duchy of Warsaw, and grew up in Warsaw, which after 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising.""",
    """The most populated city in the United States is which city?""": """New York—often called New York City or the City of New York to distinguish it from the State of New York, of which it is a part—is the most populous city in the United States and the center of the New York metropolitan area, the premier gateway for legal immigration to the United States and one of the most populous urban agglomerations in the world. A global power city, New York exerts a significant impact upon commerce, finance, media, art, fashion, research, technology, education, and entertainment, its fast pace defining the term New York minute. Home to the headquarters of the United Nations, New York is an important center for international diplomacy and has been described as the cultural and financial capital of the world."""
}

selected_text = st.selectbox('Select an Example:', list(examples.keys()))
st.subheader('Try it yourself!')
custom_input_question = st.text_input('Create a question')
custom_input_context = st.text_input("Create it's context")

custom_examples = {}

st.subheader('Selected Text')

if custom_input_question and custom_input_context:
    custom_examples[custom_input_question] = custom_input_context
    selected_text = [f'"""{next(iter(custom_examples))}""" """context:{custom_examples[next(iter(custom_examples))]}"""']
    st.markdown(f"**Text:** {custom_input_question}")
    st.markdown(f"**Context:** {custom_input_context}")
elif selected_text:
    st.markdown(f"**Text:** {selected_text}")
    st.markdown(f"**Context:** {examples[selected_text]}")
    selected_text = [f'"""{selected_text}""" """context:{examples[selected_text]}"""']

Pipeline = create_pipeline(model)
output = fit_data(Pipeline, selected_text)

st.subheader('Prediction')
st.write(output)
