import streamlit as st 
st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Speech Recognition with whisper',  # String or None. Strings get appended with "• Streamlit". 
    page_icon='../../resources/favicon.png',  # String, anything supported by st.image, or None.
)
import pandas as pd
import numpy as np
import os
import json
import base64
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
import streamlit_apps_config as config

########## To Remove the Main Menu Hamburger ########

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

########## Side Bar ########

##-- Hrishabh Digaari: FOR SOLVING THE ISSUE OF INTERMITTENT IMAGES & LOGO-----------------------------------------------------
import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url,image_size):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    if image_size == None:
        html_code = f'''
            <a href="{target_url}">
                <img height="90%" width="90%" src="data:image/{img_format};base64,{bin_str}" />
            </a>'''
    elif image_size >=300:
        html_code = f'''
            <a href="{target_url}">
                <img height={image_size} width={image_size} src="data:image/{img_format};base64,{bin_str}" />
            </a>'''
    return html_code


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

## for audio files 
def get_audio_with_href(local_audio_path, target_url):
    audio_format = os.path.splitext(local_audio_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_audio_path)
    html_code = f'''
        <a href="{target_url}">
            <audio controls src="data:audio/{audio_format};base64,{bin_str}"></audio>
        </a>'''
    return html_code


logo_html = get_img_with_href('../../resources/jsl-logo.png', 'https://www.johnsnowlabs.com/', image_size=None)
st.sidebar.markdown(logo_html, unsafe_allow_html=True)

### loading description file
descriptions = pd.read_json('../../streamlit_apps_descriptions.json')
descriptions = descriptions[descriptions['app'] == 'SPEECH_RECOGNITION_WITH_WHISPER']

### showing available models
model_names = descriptions['model_name'].values
st.sidebar.title("Choose the pretrained model")
selected_model = st.sidebar.selectbox("", list(model_names))#input_files_list)


## displaying selected model's output
app_title = descriptions[descriptions['model_name'] == selected_model]['title'].iloc[0]
app_description = descriptions[descriptions['model_name'] == selected_model]['description'].iloc[0]

st.title(app_title)
st.header(app_description)

AUDIO_FILE_PATH = os.path.join(os.getcwd(),'inputs/'+selected_model+"/")
RESULT_FILE_PATH = os.path.join(os.getcwd(),'outputs/'+selected_model+"/")

# script_dir = os.path.dirname(os.path.abspath(__file__))
# AUDIO_FILE_PATH = os.path.join(script_dir, 'inputs', selected_model)
# RESULT_FILE_PATH = os.path.join(script_dir, 'outputs', selected_model)

audio_files = sorted(os.listdir(AUDIO_FILE_PATH))
selected_audio = st.selectbox("Select an audio", audio_files)

selected_audio_path = os.path.join(AUDIO_FILE_PATH, selected_audio)

with open(RESULT_FILE_PATH + '/results.json', "r") as res:
    results = json.load(res)

# with open(os.path.join(RESULT_FILE_PATH, 'results.json'), "r") as res:
#     results = json.load(res)

st.subheader("Play Audio")

# Original = get_audio_with_href(selected_audio_path, '#')
Original = get_audio_with_href(AUDIO_FILE_PATH+selected_audio, '#')
st.markdown(Original, unsafe_allow_html=True)

st.subheader(f"Transcription for {selected_audio}:")
st.markdown(f'{results[selected_audio]}')

st.sidebar.title("")

try_link="""<a href="https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/audio/whisper/Automatic_Speech_Recognition_Whisper_(WhisperForCTC).ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""

st.sidebar.markdown("Try it yourself:")
st.sidebar.markdown(try_link, unsafe_allow_html=True)