---
layout: model
title: DistilBERT Sequence Classification - Urdu IMDB (distilbert_base_sequence_classifier_imdb)
author: John Snow Labs
name: distilbert_base_sequence_classifier_imdb
date: 2021-11-21
tags: [ur, urdu, sentiment, imdb, sequence_classification, open_source, distilbert]
task: Text Classification
language: ur
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

DistilBERT Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

`distilbert_base_sequence_classifier_imdb ` is a fine-tuned DistilBERT model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance. 

We used TFDistilBertForSequenceClassification to train this model and used BertForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`negative`, `positive`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_imdb_ur_3.3.3_3.0_1637503332156.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

sequenceClassifier = DistilBertForSequenceClassification \
.pretrained('distilbert_base_sequence_classifier_imdb', 'ur') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['یہ فلم واقعی اچھی تھی!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_imdb", "ur")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("یہ فلم واقعی اچھی تھی!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("ur.classify.distilbert_sequence.imdb").predict("""یہ فلم واقعی اچھی تھی!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_sequence_classifier_imdb|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|ur|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/imdb_urdu_reviews](https://huggingface.co/datasets/imdb_urdu_reviews)

## Benchmarking

```bash
precision    recall  f1-score   support

negative       0.94      0.96      0.95     24285
positive       0.96      0.95      0.95     25715

accuracy                           0.95     50000
macro avg       0.95      0.95      0.95     50000
weighted avg       0.95      0.95      0.95     50000
```