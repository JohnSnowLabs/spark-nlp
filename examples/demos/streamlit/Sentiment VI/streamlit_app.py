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

    sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_distilbert_sentiment", "vi") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class_")   
        
    nlpPipeline = Pipeline(
        stages=[
        document, 
        embeddings,
        sentimentClassifier
        ])

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
    layout="wide", page_title="Analyze sentiment in Vietnamese texts", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Analyze sentiment in Vietnamese texts")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_list = ["classifierdl_distilbert_sentiment"]
model = st.sidebar.selectbox("Choose the pretrained model", model_list, help="For more info about the models visit: https://sparknlp.org/models",)

# # Let's add the colab link for the notebook.

link= """<a href="https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [
    "Chủ shop nch cực nhiệt tình và dễ thương lắm luôn á, lại còn tự tay viết thiệp và tặng kèm quà nữa, hihi.",
    "Cho 3 sao vì đã nói trước là giúp mình cắt dây đeo là 6 6 rồi nhưng không cắt.",
    "Dày ko y hình vì nếu y hình thì chữ nike phải may bằng kim tuyến mình check mã thì ko có mặt hàng này dù rất buồn nhưng cx đã nhận hàng rồi và ko biết phải làm sao😑😑.",
    "Mặc dù có chút trục trặc về việc giao hàng nhưng chủ shop gọi điện nói chuyện rất lịch sự nên cũng cho qua được.",
    "Chất lượng sản phẩm rất kém Shop làm ăn ko uy tín gửi hàng ko đúng hình.",
    "Chất lượng sản phẩm tốt Đóng gói sản phẩm chắc chắn Thời gian giao hàng nhanh chỉ tiếc dây xanh gắn dây kéo bị ra màu nên e phải cắt bỏ nhưng sản phẩn khá ổn hợp với giá tiền."
]

st.subheader("This model identifies Positive or Negative sentiments in Vietnamese texts.")

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
