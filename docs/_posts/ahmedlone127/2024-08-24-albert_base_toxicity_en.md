---
layout: model
title: ALBERT Sequence Classification Base - Toxicity (albert_base_toxicity)
author: John Snow Labs
name: albert_base_toxicity
date: 2024-08-24
tags: [sequence_classification, albert, openvino, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: openvino
annotator: AlbertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“
        
        ALBERT Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

albert_base_sequence_classifier_imdb is a fine-tuned ALBERT model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance.

We used TFAlbertForSequenceClassification to train this model and used AlbertForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_toxicity_en_5.4.2_3.0_1724533542430.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_base_toxicity_en_5.4.2_3.0_1724533542430.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler() .setInputCol('text') .setOutputCol('document')

tokenizer = Tokenizer() .setInputCols(['document']) .setOutputCol('token')

sequenceClassifier = AlbertForSequenceClassification .pretrained('albert_base_toxicity', 'en') .setInputCols(['token', 'document']) .setOutputCol('class') .setCaseSensitive(False) .setMaxSentenceLength(512)

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

val tokenClassifier = AlbertForSequenceClassification.pretrained("albert_base_toxicity", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(false)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_toxicity|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|en|
|Size:|44.2 MB|
|Case sensitive:|true|