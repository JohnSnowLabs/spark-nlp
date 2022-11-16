---
layout: model
title: News Classifier of Turkish text
author: John Snow Labs
name: classifierdl_bert_news
date: 2021-05-03
tags: [tr, news, classifier, open_source]
task: Text Classification
language: tr
edition: Spark NLP 3.0.2
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify Turkish news texts

## Predicted Entities

`kultur`, `saglik`, `ekonomi`, `teknoloji`, `siyaset`, `spor`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_tr_3.0.2_3.0_1620040285456.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

embeddings = BertSentenceEmbeddings\
.pretrained('labse', 'xx') \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")

document_classifier = ClassifierDLModel.pretrained("classifierdl_bert_news", "tr") \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

nlpPipeline = Pipeline(stages=[document, embeddings, document_classifier])
light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.annotate('Bonservisi elinde olan Milli oyuncu, yeni takımıyla el sıkıştı.')
```
```scala
val document = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val embeddings = BertSentenceEmbeddings
.pretrained("labse", "xx") 
.setInputCols("document")
.setOutputCol("sentence_embeddings")

val document_classifier = ClassifierDLModel.pretrained("classifierdl_bert_news", "tr") 
.setInputCols(Array("document", "sentence_embeddings")) 
.setOutputCol("class")

val nlpPipeline = new Pipeline().setStages(Array(document, embeddings, document_classifier))
val light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([['']]).toDF("text")))
val result = light_pipeline.annotate("Bonservisi elinde olan Milli oyuncu, yeni takımıyla el sıkıştı".)
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.classify.news").predict("""Bonservisi elinde olan Milli oyuncu, yeni takımıyla el sıkıştı.""")
```

</div>

## Results

```bash
["spor"]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_news|
|Compatibility:|Spark NLP 3.0.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|tr|
|Dependencies:|labse_BERT|

## Data Source

Trained on a custom dataset with multi-lingual Bert Embeddings `labse`.

## Benchmarking

```bash
precision    recall  f1-score   support

ekonomi       0.88      0.86      0.87       263
kultur       0.93      0.96      0.94       277
saglik       0.95      0.96      0.95       273
siyaset       0.89      0.91      0.90       257
spor       0.97      0.97      0.97       279
teknoloji       0.94      0.88      0.91       250

accuracy                           0.93      1599
macro avg       0.93      0.92      0.93      1599
weighted avg       0.93      0.93      0.93      1599
```