import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import streamlit_apps_config as config
import base64

st.set_page_config(
    layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Text Summarization with BART',  # String or None. Strings get appended with "• Streamlit". 
    # page_icon='../../resources/favicon.png',  # String, anything supported by st.image, or None.
)

########## To Remove the Main Menu Hamburger ########

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

logo_html = get_img_with_href('../../resources/jsl-logo.png', 'https://www.johnsnowlabs.com/')
st.sidebar.markdown(logo_html, unsafe_allow_html=True)

### loading description file
descriptions = pd.read_json('../../streamlit_apps_descriptions.json')
descriptions = descriptions[descriptions['app'] == 'TEXT_SUMMARIZATION_WITH_BART']

### showing available models
model_names = descriptions['model_name'].values

task_list = ['summarization']

task_description = {
                    'summarization': "Summarize text into a shorter representation."
                }

desc_to_task = {v:k for k,v in task_description.items()}

# Select task selectbox
st.sidebar.title("Select Taks:")

# Description is better for user
task = st.sidebar.selectbox("Tasks", list(task_description.keys()))

# Choose model selectbox
model = st.sidebar.selectbox("Model", ['distilbart_xsum_12_6'])

## displaying selected model's output
app_title = descriptions[descriptions['model_name'] == model]['title'].iloc[0]
app_description = descriptions[descriptions['model_name'] == model]['description'].iloc[0]

### Creating Directory Structure

TEXT_FILE_PATH = os.path.join(os.getcwd(),'inputs', task, model)
RESULT_FILE_PATH = os.path.join(os.getcwd(),'outputs', task, model)

input_files = sorted(os.listdir(TEXT_FILE_PATH))
input_files_paths = [os.path.join(TEXT_FILE_PATH, fname) for fname in input_files]
result_files_paths = [os.path.join(RESULT_FILE_PATH, fname.split('.')[0]+'.txt') for fname in input_files]

title_text_mapping_dict={}
title_json_mapping_dict={}
for index, file_path in enumerate(input_files_paths):
    lines = open(file_path, 'r', encoding='utf-8').readlines()
    title = lines[0]
    text = '\n'.join(lines[1:])
    title_text_mapping_dict[title] = text
    title_json_mapping_dict[title] = result_files_paths[index]


######## Main Page ################

#st.title('Spark NLP Sentiment Analysis')
st.title(app_title)

st.markdown("<h2>"+app_description+"</h2>", unsafe_allow_html=True)
#st.subheader("")
st.markdown("<h3> Task description: "+task_description[task]+"</h3>", unsafe_allow_html=True)
st.subheader("")
selected_title = None
selected_title = st.selectbox("Select an example", list(title_text_mapping_dict.keys()))
selected_text = title_text_mapping_dict[selected_title]

selected_file = title_json_mapping_dict[selected_title]
with open(selected_file, 'r', encoding='utf-8') as f_:
    res_text = f_.read()


st.subheader('Text')
st.write(config.HTML_WRAPPER.format(selected_text), unsafe_allow_html=True)

st.subheader("Summary")
st.write(config.HTML_WRAPPER.format(res_text), unsafe_allow_html=True)


try_link="""<a href="https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-summarization/Text_Summarization_with_BART.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Try it yourself:')
st.sidebar.markdown(try_link, unsafe_allow_html=True)
