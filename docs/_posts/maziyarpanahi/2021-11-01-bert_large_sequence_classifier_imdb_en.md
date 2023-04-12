---
layout: model
title: BERT Sequence Classification Large - IMDB (bert_large_sequence_classifier_imdb)
author: John Snow Labs
name: bert_large_sequence_classifier_imdb
date: 2021-11-01
tags: [bert, large, en, english, sentiment, open_source, sequence_classification, imdb]
task: Text Classification
language: en
nav_key: models
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BERT Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

`bert_large_sequence_classifier_imdb ` is a fine-tuned BERT model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance.

We used TFBertForSequenceClassification to train this model and used BertForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`neg`, `pos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_sequence_classifier_imdb_en_3.3.2_3.0_1635774771318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_large_sequence_classifier_imdb_en_3.3.2_3.0_1635774771318.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification \
.pretrained('bert_large_sequence_classifier_imdb', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
sequenceClassifier
])

example = spark.createDataFrame([['I really liked that movie!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_large_sequence_classifier_imdb", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.bert_sequence.imdb_large").predict("""I really liked that movie!""")
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
|Model Name:|bert_large_sequence_classifier_imdb|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)

## Benchmarking

```bash
precision    recall  f1-score   support

neg       0.94      0.95      0.94     12359
pos       0.95      0.94      0.94     12641

accuracy                           0.94     25000
macro avg       0.94      0.94      0.94     25000
weighted avg       0.94      0.94      0.94     25000

```
