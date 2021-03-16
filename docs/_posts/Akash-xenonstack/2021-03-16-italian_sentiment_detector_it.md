---
layout: model
title: Sentiment Analysis for italian language
author: John Snow Labs
name: italian_sentiment_detector
date: 2021-03-16
tags: [it, open_source]
task: Sentiment Analysis
language: it
edition: Spark NLP 2.7.3
spark_version: 2.4
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Model is trained to give sentiment from Italian texts.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/models/Akash-xenonstack_italian_sentiment_detector_it_2.7.3_2.4_1615879121959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Call the model using PretainedPipeline( name = "italian_sentiment_detector", disk_location= "path to downloaded model")

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import pandas as pd
from sklearn.metrics import classification_report

# starting spark session
spark = sparknlp.start()

# Training Dataset
trainDataset = spark.read \
      .option("header", True) \
      .csv("..data/training_data/training_sentiment_italian.csv")

# Constructing pipeline stages

# Document Assembler
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

# Sentence Detector
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# Tokenizer
tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Normalizer
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normal")

# Lemmatizer
lemmatizer = Lemmatizer() \
    .setInputCols(["normal"]) \
    .setOutputCol("lemma") \
    .setDictionary(
    path="../data/training_data/lemma_italian.txt",
    read_as="TEXT",
    key_delimiter="\\s+",
    value_delimiter="->"
)

# Token Assembler
token_assembler = TokenAssembler() \
    .setInputCols(["sentence", "lemma"]) \
    .setOutputCol("document")

# Embeddings
use = UniversalSentenceEncoder.pretrained("tfhub_use_xling_many", "xx") \
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

# SentimentDLApproach
sentimentdl = SentimentDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("label")\
  .setMaxEpochs(5)\
  .setEnableOutputLogs(True)

# Compbininig stages into one single pipeline
pipeline = Pipeline(
    stages = [
        document,
        sentence_detector,
        tokenizer,
        normalizer,
        lemmatizer,
        token_assembler,
        use,
        sentimentdl
    ])
pipeline_model = pipeline.fit(trainDataset)

# Saving model
pipeline_model.save("model/italian_sentiment_detector")
```

</div>

## Results

```bash
Model is giving 69% accuracy on random italian tweets.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|italian_sentiment_detector|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Open Source|
|Edition:|Community|
|Language:|it|

## Included Models

word embeddings -Universal Sentence Encoder XLING Many( tfhub_use_xling_many)