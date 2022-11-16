---
layout: model
title: DistilBERT Sequence Classification Base - AG News (distilbert_base_sequence_classifier_ag_news)
author: John Snow Labs
name: distilbert_base_sequence_classifier_ag_news
date: 2021-11-21
tags: [sequence_classification, ag_news, distilbert, en, english, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

DistilBERT Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

`distilbert_base_sequence_classifier_ag_news ` is a fine-tuned DistilBERT model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance. 

We used TFDistilBertForSequenceClassification to train this model and used BertForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`Business`, `Sci/Tech`, `Sports`, `World`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_ag_news_en_3.3.3_3.0_1637503060617.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('distilbert_base_sequence_classifier_ag_news', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_ag_news", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distilbert_sequence.ag_news").predict("""Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_sequence_classifier_ag_news|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)

## Benchmarking

```bash
precision    recall  f1-score   support

Business       0.90      0.90      0.90      1904
Sci/Tech       0.91      0.90      0.91      1935
Sports       0.99      0.98      0.98      1921
World       0.92      0.95      0.94      1840

accuracy                           0.93      7600
macro avg       0.93      0.93      0.93      7600
weighted avg       0.93      0.93      0.93      7600
```
