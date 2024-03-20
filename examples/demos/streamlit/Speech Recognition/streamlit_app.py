############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os
import librosa
import pandas as pd

from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import *
import pyspark.sql.functions as F

spark = sparknlp.start()

def create_pipeline(model):
    
    audio_assembler = AudioAssembler() \
        .setInputCol("audio_content") \
        .setOutputCol("audio_assembler")

    speech_to_text = Wav2Vec2ForCTC \
        .pretrained(model)\
        .setInputCols("audio_assembler") \
        .setOutputCol("text")

    pipeline = Pipeline(stages=[
    audio_assembler,
    speech_to_text
    ])

    return pipeline

def fit_data(pipeline, fed_data):

    data,sampleing_rate = librosa.load(fed_data, sr=16000)
    data=[float(x) for x in data]
    schema = StructType([StructField("audio_content", ArrayType(FloatType())),
                        StructField("sampling_rate", LongType())])

    df = pd.DataFrame({
        "audio_content":[data],
        "sampling_rate":[sampleing_rate]
    })

    spark_df = spark.createDataFrame(df, schema)
    pipelineDF = pipeline.fit(spark_df).transform(spark_df)

    return pipelineDF.select("text.result")

def save_uploadedfile(uploadedfile, path):
    filepath = os.path.join(path, uploadedfile.name)
    with open(filepath, "wb") as f:
        if hasattr(uploadedfile, 'getbuffer'):
            f.write(uploadedfile.getbuffer())
        else:
            f.write(uploadedfile.read())
        
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Speech Recognition", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Speech Recognition")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)
model_list = ["asr_wav2vec2_base_100h_13K_steps","asr_wav2vec2_base_100h_ngram","asr_wav2vec2_base_100h_by_facebook","asr_wav2vec2_base_100h_test","asr_wav2vec2_base_960h"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link = """<a href="https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/open-source-nlp/17.0.Automatic_Speech_Recognition_Wav2Vec2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

st.subheader("This demo transcribes audio files into texts using advanced speech recognition models.")

AUDIO_FILE_PATH = f"/content/Speech Recognition/inputs"
audio_files = sorted(os.listdir(AUDIO_FILE_PATH))

select_audio = st.selectbox("Select an audio", audio_files)
# Creating a simplified Python list of audio file types
audio_file_types = ["mp3", "flac", "wav", "aac", "ogg", "aiff", "wma", "m4a", "ape", "dsf", "dff", "midi", "mid", "opus", "amr"]
uploadedfile = st.file_uploader("Try it for yourself!", type=audio_file_types)

if uploadedfile:
    selected_audio = f"{AUDIO_FILE_PATH}/{uploadedfile.name}"
    save_uploadedfile(uploadedfile, AUDIO_FILE_PATH)
elif select_audio:
    selected_audio = f"{AUDIO_FILE_PATH}/{select_audio}"

st.subheader("Play Audio")

audio_file = open(selected_audio, 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes)

st.subheader(f"Transcription for {select_audio}:")

Pipeline = create_pipeline(model)
output = fit_data(Pipeline, selected_audio)

st.text(output.first().result[0].strip())

