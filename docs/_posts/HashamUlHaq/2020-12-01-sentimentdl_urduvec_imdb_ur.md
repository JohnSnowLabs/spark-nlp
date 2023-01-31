---
layout: model
title: Sentiment Analysis for Urdu (IMDB Review dataset)
author: John Snow Labs
name: sentimentdl_urduvec_imdb
date: 2020-12-01
task: Sentiment Analysis
language: ur
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [sentiment, ur, open_source]
supported: true
annotator: SentimentDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Analyse sentiment in reviews by classifying them as ``positive``, ``negative`` or ``neutral``. This model is trained using ``urduvec_140M_300d`` word embeddings. The word embeddings are then converted to sentence embeddings before feeding to the sentiment classifier which uses a DL architecture to classify sentences.

{:.h2_title}
## Predicted Entities

``positive`` , ``negative`` , ``neutral``.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_urduvec_imdb_ur_2.7.0_2.4_1606817135630.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentimentdl_urduvec_imdb_ur_2.7.0_2.4_1606817135630.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, SentenceEmbeddings.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
word_embeddings = WordEmbeddingsModel()\
.pretrained('urduvec_140M_300d', 'ur')\
.setInputCols(["document",'token'])\
.setOutputCol("word_embeddings")
embeddings = SentenceEmbeddings() \
.setInputCols(["document", "word_embeddings"]) \
.setOutputCol("sentence_embeddings") \
.setPoolingStrategy("AVERAGE")
classifier = SentimentDLModel.pretrained('sentimentdl_urduvec_imdb', 'ur' )\
.setInputCols(['document', 'token', 'sentence_embeddings']).setOutputCol('sentiment')
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, sentence_embeddings, classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate(["مجھے واقعی یہ شو سند ہے۔ یہی وجہ ہے کہ مجھے حال ہی میں یہ جان کر مایوسی ہوئی ہے کہ جارج لوپیز ایک ",
"بالکل بھی اچھ ،ی کام نہیں کیا گیا ، پوری فلم صرف گرڈج تھی اور کہیں بھی بے ترتیب لوگوں کو ہلاک نہیں"])
```
```scala
...
val word_embeddings = WordEmbeddingsModel()
.pretrained('urduvec_140M_300d', 'ur')
.setInputCols(Array("document",'token'))
.setOutputCol("word_embeddings")
val embeddings = SentenceEmbeddings() 
.setInputCols(Array("document", "word_embeddings")) 
.setOutputCol("sentence_embeddings")
.setPoolingStrategy("AVERAGE")
val classifier = SentimentDLModel.pretrained('sentimentdl_urduvec_imdb', 'ur' )
.setInputCols(Array('document', 'token', 'sentence_embeddings')).setOutputCol('sentiment')
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, sentence_embeddings, classifier))
val result = pipeline.fit(Seq.empty["مجھے واقعی یہ شو سند ہے۔ یہی وجہ ہے کہ مجھے حال ہی میں یہ جان کر مایوسی ہوئی ہے کہ جارج لوپیز ایک ",
"بالکل بھی اچھ ،ی کام نہیں کیا گیا ، پوری فلم صرف گرڈج تھی اور کہیں بھی بے ترتیب لوگوں کو ہلاک نہیں"].toDS.toDF("text")).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["مجھے واقعی یہ شو سند ہے۔ یہی وجہ ہے کہ مجھے حال ہی میں یہ جان کر مایوسی ہوئی ہے کہ جارج لوپیز ایک ", "بالکل بھی اچھ ،ی کام نہیں کیا گیا ، پوری فلم صرف گرڈج تھی اور کہیں بھی بے ترتیب لوگوں کو ہلاک نہیں"]
urdusent_df = nlu.load('ur.sentiment').predict(text, output_level='sentence')
urdusent_df
```

</div>

## Results

```bash

|    | document                                                                                                 | sentiment     |
|---:|---------------------------------------------------------------------------------------------------------:|--------------:|
|  0 |مجھے واقعی یہ شو سند ہے۔ یہی وجہ ہے کہ مجھے حال ہی میں یہ جان کر مایوسی ہوئی ہے کہ جارج لوپیز ایک  | positive      |
|  1 |بالکل بھی اچھ ،ی کام نہیں کیا گیا ، پوری فلم صرف گرڈج تھی اور کہیں بھی بے ترتیب لوگوں کو ہلاک نہیں  | negative      |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentimentdl_urduvec_imdb|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[sentiment]|
|Language:|ur|
|Dependencies:|urduvec_140M_300d|

## Data Source

This models in trained using data from https://www.kaggle.com/akkefa/imdb-dataset-of-50k-movie-translated-urdu-reviews

## Benchmarking

```bash
loss: 2428.622 - acc: 0.8181 - val_acc: 80.0
```