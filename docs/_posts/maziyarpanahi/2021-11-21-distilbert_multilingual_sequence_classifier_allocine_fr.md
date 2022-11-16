---
layout: model
title: DistilBERT Sequence Classification French - AlloCine (distilbert_multilingual_sequence_classifier_allocine)
author: John Snow Labs
name: distilbert_multilingual_sequence_classifier_allocine
date: 2021-11-21
tags: [sequence_classification, fr, french, distilbert, sentiment, open_source]
task: Text Classification
language: fr
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

`distilbert_multilingual_sequence_classifier_allocine ` is a fine-tuned DistilBERT model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance. 

We used TFDistilBertForSequenceClassification to train this model and used BertForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`neg`, `pos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_multilingual_sequence_classifier_allocine_fr_3.3.3_3.0_1637501723857.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('distilbert_multilingual_sequence_classifier_allocine', 'fr') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['j'ai bien aime le film harry potter!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_multilingual_sequence_classifier_allocine", "fr")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("j'ai bien aime le film harry potter!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.classify.distilbert_sequence.allocine").predict("""Put your text here.""")
```

</div>

## Results

```bash
* +--------------------+
* |result              |
* +--------------------+
* |[neg, neg]          |
* |[pos, pos, pos, pos]|
* +--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_multilingual_sequence_classifier_allocine|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|fr|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/allocine](https://huggingface.co/datasets/allocine)

## Benchmarking

```bash
precision    recall  f1-score   support

neg       0.95      0.96      0.95     10269
pos       0.96      0.94      0.95      9731

accuracy                           0.95     20000
macro avg       0.95      0.95      0.95     20000
weighted avg       0.95      0.95      0.95     20000
```
