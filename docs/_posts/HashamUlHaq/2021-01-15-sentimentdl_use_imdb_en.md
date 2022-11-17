---
layout: model
title: Sentiment Analysis of IMDB Reviews (sentimentdl_use_imdb)
author: John Snow Labs
name: sentimentdl_use_imdb
date: 2021-01-15
task: Sentiment Analysis
language: en
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, en, sentiment]
supported: true
annotator: SentimentDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify IMDB reviews in negative and positive categories using `Universal Sentence Encoder`.

## Predicted Entities

`neg`, `pos`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_use_imdb_en_2.7.0_2.4_1610715247685.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

use = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")

classifier = SentimentDLModel().pretrained('sentimentdl_use_imdb')\
.setInputCols(["sentence_embeddings"])\
.setOutputCol("sentiment")

nlp_pipeline = Pipeline(stages=[document_assembler,
use,
classifier
])

l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate('Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!')

```




{:.nlu-block}
```python
import nlu
nlu.load("en.sentiment.imdb.use.dl").predict("""Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!""")
```

</div>

## Results

```bash
|    | document                                                                                                 | sentiment     |
|---:|---------------------------------------------------------------------------------------------------------:|--------------:|
|    | Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the  |               |
|  0 | film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music   | positive      |
|    | was rad! Horror and sword fight freaks,buy this movie now!                                               |               |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentimentdl_use_imdb|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[sentiment]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

This model is trained on data from https://ai.stanford.edu/~amaas/data/sentiment/

## Benchmarking

```bash
precision    recall  f1-score   support

neg       0.88      0.82      0.85     12500
pos       0.84      0.88      0.86     12500

accuracy                           0.85     25000
macro avg       0.86      0.86      0.85     25000
weighted avg       0.86      0.85      0.85     25000
```