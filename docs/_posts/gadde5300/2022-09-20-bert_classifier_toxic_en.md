---
layout: model
title: English BertForSequenceClassification Cased model (from unitary)
author: John Snow Labs
name: bert_classifier_toxic
date: 2022-09-20
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `toxic-bert` is a English model originally trained by `unitary`.

## Predicted Entities

`obscene`, `insult`, `severe_toxic`, `identity_hate`, `threat`, `toxic`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_toxic_en_4.2.0_3.0_1663668813784.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_toxic","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_toxic","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_toxic|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/unitary/toxic-bert
- https://github.com/unitaryai/detoxify/issues/15
- https://github.com/unitaryai/detoxify
- https://laurahanu.github.io/
- https://www.unitary.ai/
- https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
- https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification
- https://homes.cs.washington.edu/~msap/pdfs/sap2019risk.pdf
- https://arxiv.org/pdf/1703.04009.pdf%201.pdf
- https://arxiv.org/pdf/1905.12516.pdf
- https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
- https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
- https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api
- https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation