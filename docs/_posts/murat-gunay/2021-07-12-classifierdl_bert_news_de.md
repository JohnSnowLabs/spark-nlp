---
layout: model
title: News Classifier of German text
author: John Snow Labs
name: classifierdl_bert_news
date: 2021-07-12
tags: [de, news, classifier, open_source, german]
task: Text Classification
language: de
edition: Spark NLP 3.1.1
spark_version: 2.4
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify German texts of news

## Predicted Entities

`Inland`, `International`, `Panorama`, `Sport`, `Web`, `Wirtschaft`, `Wissenschaft`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_DE_NEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_DE_NEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_de_3.1.1_2.4_1626079085859.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_de_3.1.1_2.4_1626079085859.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

embeddings = BertSentenceEmbeddings\
    .pretrained('sent_bert_multi_cased', 'xx') \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

document_classifier = ClassifierDLModel.pretrained("classifierdl_bert_news", "de") \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("class")

nlpPipeline = Pipeline(stages=[document, embeddings, document_classifier])

light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

result = light_pipeline.annotate("Niki Lauda in einem McLaren MP 4/2 TAG Turbo. Mit diesem Gefährt sicherte sich der Österreicher 1984 seinen dritten Weltmeistertitel, einen halben (!)")
```
```scala
val document = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val embeddings = BertSentenceEmbeddings
    .pretrained("sent_bert_multi_cased", "xx") 
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")

val document_classifier = ClassifierDLModel.pretrained("classifierdl_bert_news", "de") 
    .setInputCols(Array("document", "sentence_embeddings")) 
    .setOutputCol("class")

val nlpPipeline = new Pipeline().setStages(Array(document, embeddings, document_classifier))

val data = Seq("""Niki Lauda in einem McLaren MP 4/2 TAG Turbo. Mit diesem Gefährt sicherte sich der Österreicher 1984 seinen dritten Weltmeistertitel, einen halben (!)""").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.classify.news").predict("""Niki Lauda in einem McLaren MP 4/2 TAG Turbo. Mit diesem Gefährt sicherte sich der Österreicher 1984 seinen dritten Weltmeistertitel, einen halben (!)""")
```

</div>

## Results

```bash
['Sport']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_news|
|Compatibility:|Spark NLP 3.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|de|

## Data Source

Trained on a custom dataset with multi-lingual Bert Sentence Embeddings.

## Benchmarking

```bash
label           precision   recall  f1-score   support
Inland          0.78        0.81      0.79       102
International   0.80        0.89      0.84       151
Panorama        0.84        0.70      0.76       168
Sport           0.98        0.99      0.98       120
Web             0.93        0.90      0.91       168
Wirtschaft      0.74        0.83      0.78       141
Wissenschaft    0.84        0.75      0.80        57
accuracy        -           -         0.84       907
macro-avg       0.84        0.84      0.84       907
weighted-avg    0.85        0.84      0.84       907
```