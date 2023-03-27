---
layout: model
title: Financial Target NER (Codalab)
author: John Snow Labs
name: finner_bert_target
date: 2023-03-27
tags: [bert, finance, ner, es, licensed, tensorflow]
task: Named Entity Recognition
language: es
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: FinanceBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This  Spanish NER model will identify the label `TARGET` from a financial statement. This model is trained from the competition - `IBERLEF 2023 Task - FinancES. Financial Targeted Sentiment Analysis in Spanish`. We have used the participation dataset which is a small subset of the main one to train this model.

## Predicted Entities

`TARGET`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_bert_target_es_1.0.0_3.0_1679942128323.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_bert_target_es_1.0.0_3.0_1679942128323.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")
  
tokenClassifier = finance.BertForTokenClassification.pretrained("finner_bert_target","es","finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

converter = finance.NerConverterInternal()\
    .setInputCols(["document", "token", "label"])\
    .setOutputCol("ner")

pipeline =  nlp.Pipeline(
    stages=[
  documentAssembler,
  tokenizer,
  tokenClassifier,
  converter
    ]
)

```

</div>

## Results

```bash
+-----------+------+
|chunk      |entity|
+-----------+------+
|Presupuesto|TARGET|
|populista  |TARGET|
+-----------+------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_bert_target|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|406.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://codalab.lisn.upsaclay.fr/competitions/10052#learn_the_details

## Benchmarking

```bash
 
labels              precision    recall  f1-score   support
    B-TARGET       0.76      0.82      0.79       435
   micro-avg       0.76      0.82      0.79       435
   macro-avg       0.76      0.82      0.79       435
weighted-avg       0.76      0.82      0.79       435

```