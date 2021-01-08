# Spark NLP: State of the Art Natural Language Processing

[![build](https://github.com/JohnSnowLabs/spark-nlp/workflows/build/badge.svg)](https://github.com/JohnSnowLabs/spark-nlp/actions) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.johnsnowlabs.nlp/spark-nlp_2.11/badge.svg)](https://search.maven.org/artifact/com.johnsnowlabs.nlp/spark-nlp_2.11) [![PyPI version](https://badge.fury.io/py/spark-nlp.svg)](https://badge.fury.io/py/spark-nlp) [![Anaconda-Cloud](https://anaconda.org/johnsnowlabs/spark-nlp/badges/version.svg)](https://anaconda.org/JohnSnowLabs/spark-nlp) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE)

Spark NLP is a Natural Language Processing library built on top of Apache Spark ML. It provides **simple**, **performant** & **accurate** NLP annotations for machine learning pipelines that **scale** easily in a distributed environment. Spark NLP comes with **1100+** pretrained **pipelines** and **models** in more than **192+** languages. It supports state-of-the-art transformers such as **BERT**, **XLNet**, **ELMO**, **ALBERT**, and **Universal Sentence Encoder** that can be used seamlessly in a cluster. It also offers Tokenization, Word Segmentation, Part-of-Speech Tagging, Named Entity Recognition, Dependency Parsing, Spell Checking, Multi-class Text Classification, Multi-class Sentiment Analysis, Machine Translation (+180 languages), Summarization and Question Answering **(Google T5)**, and many more [NLP tasks](#features).

## Project's website

Take a look at our official Spark NLP page: [http://nlp.johnsnowlabs.com/](http://nlp.johnsnowlabs.com/) for user documentation and examples

## Community support

- [Slack](https://spark-nlp.slack.com/join/shared_invite/zt-j5ttxh0z-Fn3lQSG1Z0KpOs_SRxjdyw#/) For live discussion with the Spark NLP community and the team
- [GitHub](https://github.com/JohnSnowLabs/spark-nlp) Bug reports, feature requests, and contributions
- [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!
- [Medium](https://medium.com/spark-nlp) Spark NLP articles
- [YouTube](https://www.youtube.com/channel/UCmFOjlpYEhxf_wJUDuz6xxQ/videos) Spark NLP video tutorials

## Features

* Tokenization
* Trainable Word Segmentation
* Stop Words Removal
* Token Normalizer
* Document Normalizer
* Stemmer
* Lemmatizer
* NGrams
* Regex Matching
* Text Matching
* Chunking
* Date Matcher
* Sentence Detector
* Deep Sentence Detector (Deep learning)
* Dependency parsing (Labeled/unlabeled)
* Part-of-speech tagging
* Sentiment Detection (ML models)
* Spell Checker (ML and DL models)
* Word Embeddings (GloVe and Word2Vec)
* BERT Embeddings (TF Hub models)
* ELMO Embeddings (TF Hub models)
* ALBERT Embeddings (TF Hub models)
* XLNet Embeddings
* Universal Sentence Encoder (TF Hub models)
* BERT Sentence Embeddings (42 TF Hub models)
* Sentence Embeddings
* Chunk Embeddings
* Unsupervised keywords extraction
* Language Detection & Identification (up to 375 languages)
* Multi-class Sentiment analysis (Deep learning)
* Multi-label Sentiment analysis (Deep learning)
* Multi-class Text Classification (Deep learning)
* Neural Machine Translation
* Text-To-Text Transfer Transformer (Google T5)
* Named entity recognition (Deep learning)
* Easy TensorFlow integration
* GPU Support
* Full integration with Spark ML functions
* +710 pre-trained models in +192 languages!
* +450 pre-trained pipelines in +192 languages!
* Multi-lingual NER models: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Hewbrew, Italian, Japanese, Korean, Norwegian, Persian, Polish, Portuguese, Russian, Spanish, Swedish, and Urdu.

## Quick Start

This is a quick example of how to use Spark NLP pre-trained pipeline in Python and PySpark:

```sh
$ java -version
# should be Java 8 (Oracle or OpenJDK)
$ conda create -n sparknlp python=3.6 -y
$ conda activate sparknlp
$ pip install spark-nlp pyspark==2.4.7
```

In Python console or Jupyter `Python3` kernel:

```python
# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Start Spark Session with Spark NLP
# start() functions has two parameters: gpu and spark23
# sparknlp.start(gpu=True) will start the session with GPU support
# sparknlp.start(sparrk23=True) is when you have Apache Spark 2.3.x installed
spark = sparknlp.start()

# Download a pre-trained pipeline
pipeline = PretrainedPipeline('explain_document_dl', lang='en')

# Your testing dataset
text = """
The Mona Lisa is a 16th century oil painting created by Leonardo.
It's held at the Louvre in Paris.
"""

# Annotate your testing dataset
result = pipeline.annotate(text)

# What's in the pipeline
list(result.keys())
Output: ['entities', 'stem', 'checked', 'lemma', 'document',
'pos', 'token', 'ner', 'embeddings', 'sentence']

# Check the results
result['entities']
Output: ['Mona Lisa', 'Leonardo', 'Louvre', 'Paris']
```

For more examples, you can visit our dedicated [repository](https://github.com/JohnSnowLabs/spark-nlp-workshop) to showcase all Spark NLP use cases!
