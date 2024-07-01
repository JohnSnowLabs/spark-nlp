############ IMPORTING LIBRARIES ############

# Import streamlit, Spark NLP, pyspark

import streamlit as st
import sparknlp
import os

from sparknlp_display import NerVisualizer
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import StringType, IntegerType
    
spark = sparknlp.start()

def create_pipeline(model, language):

    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    sentenceDetector = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentences')

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized")\
        .setLowercase(True)\
        .setCleanupPatterns(["[^\w\d\s]"])

    stopwords_cleaner = StopWordsCleaner()\
        .setInputCols("token")\
        .setOutputCol("removed_stopwords")\
        .setCaseSensitive(False)\

    stemmer = Stemmer() \
        .setInputCols(["token"]) \
        .setOutputCol("stem")


    lemmatizer = Lemmatizer() \
        .setInputCols(["token"]) \
        .setOutputCol("lemma") \
        .setDictionary("./AntBNC_lemmas_ver_001.txt", value_delimiter ="\t", key_delimiter = "->")

    nlpPipeline = Pipeline(stages=[documentAssembler,
                                sentenceDetector,
                                tokenizer,
                                normalizer,
                                stopwords_cleaner,
                                stemmer,
                                lemmatizer])

    return nlpPipeline

def fit_data(pipeline, data):

  empty_df = spark.createDataFrame([['']]).toDF('text')
  pipeline_model = pipeline.fit(empty_df)
  model = LightPipeline(pipeline_model)
  result = model.fullAnnotate(data)

  return result

    
############ SETTING UP THE PAGE LAYOUT ############
jsl_logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'jsl_logo_icon.png')

st.set_page_config(
    layout="wide", page_title="Typo Detector", page_icon=jsl_logo_path, initial_sidebar_state="auto"
)

st.caption("")
st.title("Typo Detector")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

### SIDEBAR CONTENT ###

st.sidebar.write("")

logo_path = next(f'{root}/{name}' for root, dirs, files in os.walk('/content') for name in files if name == 'logo.png')
logo = st.sidebar.image(logo_path, width=300)

model_name = ["SentenceDetector|Tokenizer|Stemmer|Lemmatizer|Normalizer|Stop Words Remover"]
#model = st.sidebar.selectbox("Choose the pretrained model", model_name, help="For more info about the models visit: https://sparknlp.org/models",)

st.sidebar.title("Filter Annotator Outputs")
selected_models = []
for model in model_name.split('|'):
  check = st.sidebar.checkbox(model, value=True, key=model)
  selected_models.append(check)

# # Let's add the colab link for the notebook.

link= """<a href="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" style="zoom: 1.3" alt="Open In Colab"/></a>"""
st.sidebar.title('')
st.sidebar.markdown('Reference notebook:')
st.sidebar.markdown(link, unsafe_allow_html=True)

### MAIN CONTENT ###

examples = [

    "The Geneva Motor Show, the first major car show of the year, opens tomorrow with U.S. Car makers hoping to make new inroads into European markets due to the cheap dollar, automobile executives said. Ford Motor Co and General Motors Corp sell cars in Europe, where about 10.5 mln new cars a year are bought. GM also makes a few thousand in North American plants for European export.",
    "Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!",
    "Quantum computing is the use of quantum-mechanical phenomena such as superposition and entanglement to perform computation. Computers that perform quantum computations are known as quantum computers. Quantum computers are believed to be able to solve certain computational problems, such as integer factorization (which underlies RSA encryption), substantially faster than classical computers. The study of quantum computing is a subfield of quantum information science. Quantum computing began in the early 1980s, when physicist Paul Benioff proposed a quantum mechanical model of the Turing machine.",
    "Titanic is a 1997 American epic romance and disaster film directed, written, co-produced, and co-edited by James Cameron. Incorporating both historical and fictionalized aspects, it is based on accounts of the sinking of the RMS Titanic, and stars Leonardo DiCaprio and Kate Winslet as members of different social classes who fall in love aboard the ship during its ill-fated maiden voyage.",
    "William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.[9] He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.",
    """World War II (often abbreviated as WWII or WW2), also known as the Second World War, was a global war that lasted from 1939 to 1945. The vast majority of the world's countries—including all the great powers—eventually formed two opposing military alliances: the Allies and the Axis. A state of total war emerged, directly involving more than 100 million people from more than 30 countries. The major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, marked by 70 to 85 million fatalities, most of whom were civilians in the Soviet Union and China. Tens of millions of people died during the conflict due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict which included the use of terror bombing, strategic bombing and the only use of nuclear weapons in war.""",
    "Disney Channel (originally called The Disney Channel from 1983 to 1997 and commonly shortened to Disney from 1997 to 2002) is an American pay television channel that serves as the flagship property of owner Disney Channels Worldwide unit of the Walt Disney Television subsidiary of The Walt Disney Company. Disney Channel's programming consists of original first-run television series, theatrically released and original made-for-TV movies and select other third-party programming. Disney Channel – which formerly operated as a premium service – originally marketed its programs towards families during the 1980s, and later at younger children by the 2000s.",
    "For several hundred thousand years, the Sahara has alternated between desert and savanna grassland in a 20,000 year cycle[8] caused by the precession of the Earth's axis as it rotates around the Sun, which changes the location of the North African Monsoon. The area is next expected to become green in about 15,000 years (17,000 ACE).",
    "Elon Musk is an engineer, industrial designer, technology entrepreneur and philanthropist. He is a citizen of South Africa, Canada, and the United States. He is the founder, CEO and chief engineer/designer of SpaceX; early investor, CEO and product architect of Tesla, Inc.; founder of The Boring Company; co-founder of Neuralink; and co-founder and initial co-chairman of OpenAI. He was elected a Fellow of the Royal Society (FRS) in 2018. In December 2016, he was ranked 21st on the Forbes list of The World's Most Powerful People, and was ranked joint-first on the Forbes list of the Most Innovative Leaders of 2019. A self-made billionaire, as of June 2020 his net worth was estimated at $38.8 billion and he is listed by Forbes as the 31st-richest person in the world. He is the longest tenured CEO of any automotive manufacturer globally.",
    "Born and raised in the Austrian Empire, Tesla studied engineering and physics in the 1870s without receiving a degree, and gained practical experience in the early 1880s working in telephony and at Continental Edison in the new electric power industry. In 1884 he emigrated to the United States, where he became a naturalized citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own. With the help of partners to finance and market his ideas, Tesla set up laboratories and companies in New York to develop a range of electrical and mechanical devices. His alternating current (AC) induction motor and related polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and became the cornerstone of the polyphase system which that company eventually marketed."

]

st.subheader("Split and clean text")

selected_text = st.selectbox("Select an example", examples)

custom_input = st.text_input("Try it for yourself!")

if custom_input:
    selected_text = custom_input
elif selected_text:
    selected_text = selected_text

st.subheader('Selected Text')
st.write(selected_text)

Pipeline = create_pipeline(model, language)
output = fit_data(Pipeline, selected_text)

df = output.toPandas()

### Displaying sentences
if selected_models[0] is True:
  st.subheader("Detected sentences in text:")
  st.dataframe(pd.DataFrame({'sentences': np.asarray(df['sentences'].values[0])[:,3]}))

## Displaying tokens, stem, lemma
equal_columns = ['token', 'stem', 'lemma']
sbhrds = ["Basic", 'Stemmed', 'Lemmatized']
tdf = pd.DataFrame()
sbhdr = ''
for index, col in enumerate(equal_columns):
  if selected_models[index+1] is True:
    tcol_arr = np.asarray(df[col].values[0])
    tdf[col] = tcol_arr[:,3]
    sbhdr += (',\t'+sbhrds[index])
if tdf.shape[0]>=1:
  st.subheader(sbhdr[1:]+"\tTokens:")
  st.dataframe(tdf)

if selected_models[4] is True:
  tcol_arr = np.asarray(df['normalized'].values[0])[:,3]
  st.subheader("Tokens after removing punctuations:")
  st.dataframe(pd.DataFrame({'normalized':tcol_arr}))

if selected_models[5] is True:
  tcol_arr = np.asarray(df['removed_stopwords'].values[0])[:,3]
  st.subheader("Tokens after removing Stop Words:")
  st.dataframe(pd.DataFrame({'removed_stopwords':tcol_arr}))