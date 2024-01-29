import streamlit as st 
st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='CLIPForZeroShotClassification for Zero Shot Classification',  # String or None. Strings get appended with "• Streamlit". 
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

style_config = config.STYLE_CONFIG
style_config = style_config.replace('height: 500px;', 'min-height: 200px; max-height: 500px; background-color: #F1F2F6; line-height: 2.0;')
st.markdown(style_config, unsafe_allow_html=True)
root_path = config.project_path

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

logo_html = get_img_with_href('../../resources/jsl-logo.png', 'https://www.johnsnowlabs.com/', image_size=None)
st.sidebar.markdown(logo_html, unsafe_allow_html=True)


## loading logo (older version without href)
# st.sidebar.image('../../resources/jsl-logo.png', use_column_width=True)

### loading description file
descriptions = pd.read_json(rf'C:\Users\bdllh\Desktop\Spark NLP new features\streamlit_apps_descriptions.json')
descriptions = descriptions[descriptions['app'] == 'CLIPForZeroShotClassification']

### showing available models
model_names = descriptions['model_name'].values[0]
st.sidebar.title("Choose the pretrained model")
selected_model = st.sidebar.selectbox("Model", ['CLIPForZeroShotClassification'])#input_files_list)

## displaying selected model's output
# app_title = descriptions[descriptions['model_name'] == selected_model]['title'].iloc[0]
# nc = descriptions[descriptions['model_name'] == selected_model]['class_number']
# app_description = descriptions[descriptions['model_name'] == selected_model]['description'].iloc[0]

app_title = descriptions[descriptions['model_name'] == model_names]['title'].iloc[0]
app_description = descriptions[descriptions['model_name'] == model_names]['description'].iloc[0]

st.title(app_title)
st.subheader('Model Name: '+model_names)
st.header(app_description)

IMAGE_FILE_PATH = os.path.join(os.getcwd(),'inputs/'+model_names+"/")
RESULT_FILE_PATH = os.path.join(os.getcwd(),'outputs/'+model_names+"/")

# input_files = sorted(os.listdir(IMAGE_FILE_PATH))
# input_files_paths = [os.path.join(IMAGE_FILE_PATH, fname) for fname in input_files]
# result_files_paths = [os.path.join(RESULT_FILE_PATH, fname.split('.')[0]+'.json') for fname in input_files]

image_files = sorted([file for file in os.listdir(IMAGE_FILE_PATH) if file.split('.')[-1]=='png' or file.split('.')[-1]=='jpg' or file.split('.')[-1]=='JPEG' or file.split('.')[-1]=='jpeg'])

selected_image = st.sidebar.selectbox("Select an image", image_files)

image_size = st.sidebar.slider('Image Size', 600, 1000, value=600, step = 100)

#result = ''.join(open(RESULT_FILE_PATH+selected_image+'_result.jpg', 'r').readlines())

with open(RESULT_FILE_PATH + 'results.json', "r") as res:
    results = json.load(res)

st.subheader('Classified Image')
Original = get_img_with_href(IMAGE_FILE_PATH+selected_image, '#', image_size)
st.markdown(Original, unsafe_allow_html=True)

# def run_pipeline(selected_model): 
# dynamically obtain the number of class, it needs to import more libraries above to run
#     image_assembler = ImageAssembler() \
#         .setInputCol("image") \
#         .setOutputCol("image_assembler")

#     image_captioning = CLIPForZeroShotClassification \
#         .pretrained(selected_model, "en") \
#         .setInputCols(["image_assembler"]) \
#         .setOutputCol("label") \
#         .setCandidateLabels(labels)

#     pipeline = Pipeline(stages=[
#         image_assembler,
#         image_captioning,
#     ])

#     return pipeline


st.subheader('Classification')
st.markdown(f'This document has been classified as  : **{results[selected_image]}**')
#st.markdown(f'Classification Confidence : **{results[selected_image][1]*100:.1f}%**')
# st.markdown(f'Number of classes in this model  : **{int(nc)}**')

st.sidebar.title("")

try_link="""<a href="https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/image/CLIPForZeroShotClassification.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""

st.sidebar.markdown("Try it yourself:")
st.sidebar.markdown(try_link, unsafe_allow_html=True)

























































# ### Creating Directory Structure

# TEXT_FILE_PATH = os.path.join(os.getcwd(),'inputs/'+selected_model)
# RESULT_FILE_PATH = os.path.join(os.getcwd(),'outputs/'+selected_model)

# input_files = sorted(os.listdir(TEXT_FILE_PATH))
# input_files_paths = [os.path.join(TEXT_FILE_PATH, fname) for fname in input_files]
# result_files_paths = [os.path.join(RESULT_FILE_PATH, fname.split('.')[0]+'.json') for fname in input_files]

# title_text_mapping_dict={}
# title_json_mapping_dict={}
# for index, file_path in enumerate(input_files_paths):
#   lines = open(file_path, 'r', encoding='utf-8').readlines()
#   title = lines[0]
#   text = lines[1]
#   title_text_mapping_dict[title] = text
#   title_json_mapping_dict[title] = result_files_paths[index]


# ######## Main Page ################

# #st.title('Spark NLP Text Classification')

# st.title(app_title)
# st.markdown("<h2>"+app_description+"</h2>", unsafe_allow_html=True)
# st.subheader("")

# #st.subheader(app_description)
# selection_text = "Pick a twitt from the below list:"
# selected_title = st.selectbox(selection_text, list(title_text_mapping_dict.keys()))
# selected_text = title_text_mapping_dict[selected_title]

# st.subheader('Full example text')
# st.write(config.HTML_WRAPPER.format(selected_text), unsafe_allow_html=True)

# selected_file = title_json_mapping_dict[selected_title]
# df = pd.read_json(selected_file)

# class_ = df.iloc[0]['pred_class'][0][3]
# Original_class = "Non-antisemitic" if class_ == "0" else "Antisemitic"
# class_ = 'Some('+str(class_)+')'
# score_ = round(float(df.iloc[0]['pred_class'][0][4][class_])*100, 2)

# result = f"This sentence has been classified as :  **{Original_class.title()}**"
# st.markdown(result)

# st.markdown("Classification Confidence: **{}%**".format(score_))

# st.title("")



# try_link="""<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/BertForSequenceClassification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
# st.sidebar.title('')
# st.sidebar.markdown('Try it yourself:')
# st.sidebar.markdown(try_link, unsafe_allow_html=True)