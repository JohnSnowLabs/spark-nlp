---
layout: model
title: Italian BertForSequenceClassification Cased model (from Hate-speech-CNERG)
author: John Snow Labs
name: bert_sequence_classifier_dehate_mono_italian
date: 2023-03-18
tags: [it, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: it
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `dehatebert-mono-italian` is a Italian model originally trained by `Hate-speech-CNERG`.

## Predicted Entities

`NON_HATE`, `HATE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_dehate_mono_italian_it_4.3.1_3.0_1679142858152.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_dehate_mono_italian_it_4.3.1_3.0_1679142858152.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_dehate_mono_italian","it") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_dehate_mono_italian","it") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_dehate_mono_italian|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|it|
|Size:|628.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-italian
- https://github.com/punyajoy/DE-LIMIT
- https://arxiv.org/abs/2004.06465