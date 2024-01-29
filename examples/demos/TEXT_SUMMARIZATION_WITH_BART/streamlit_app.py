import streamlit as st 
st.set_page_config(
    layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Text Summarization with BART',  # String or None. Strings get appended with "• Streamlit". 
    page_icon='../../resources/favicon.png',  # String, anything supported by st.image, or None.
)
import pandas as pd
import numpy as np
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
import streamlit_apps_config as config

root_path = config.project_path

style_config = config.STYLE_CONFIG
style_config = style_config.replace('height: 500px;', 'min-height: 200px; max-height: 500px; background-color: #F1F2F6; line-height: 2.0;')
st.markdown(style_config, unsafe_allow_html=True)


########## To Remove the Main Menu Hamburger ########
import base64
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

########## Side Bar ########

##-- Hrishabh Digaari: FOR SOLVING THE ISSUE OF INTERMITTENT IMAGES & LOGO-----------------------------------------------------
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img height="90%" width="90%" src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code

logo_html = get_img_with_href('../../resources/jsl-logo.png', 'https://www.johnsnowlabs.com/')
st.sidebar.markdown(logo_html, unsafe_allow_html=True)


## loading logo (older version without href)
# st.sidebar.image('../../resources/jsl-logo.png', use_column_width=True)

### loading description file
descriptions = pd.read_json('../../streamlit_apps_descriptions.json')
descriptions = descriptions[descriptions['app'] == 'TEXT_SUMMARIZATION_WITH_BART']

### showing available models
model_names = descriptions['model_name'].values

task_list = ['summarization']

task_description = {
                    'summarization': "Summarize text into a shorter representation."
                }

task_description_small = {
                    'summarization': "Summarize texts",
                }

desc_to_task = {v:k for k,v in task_description.items()}

# Select task selectbox
st.sidebar.title("Select Taks:")

# Description is better for user
task = st.sidebar.selectbox("Tasks", list(task_description.keys()))

# Back to task
#task = desc_to_task[task_desc]

# Choose model selectbox
## Two models for the 18 tasks, one model for "qa"
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
