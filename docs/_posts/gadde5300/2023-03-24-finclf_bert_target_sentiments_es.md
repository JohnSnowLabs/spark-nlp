---
layout: model
title: Financial Target Sentiments (Codalab)
author: John Snow Labs
name: finclf_bert_target_sentiments
date: 2023-03-24
tags: [bert, finance, classification, es, licensed, tensorflow, open_source]
task: Text Classification
language: es
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: FinanceBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This  Spanish Text Classifier will identify from the viewpoint of a target whether a financial statement is `positive`, `neutral` or `negative`. This model is trained from the competition - `IBERLEF 2023 Task - FinancES. Financial Targeted Sentiment Analysis in Spanish`. We have used the participation dataset which is a small subset of the main one to train this model.

## Predicted Entities

`positive`, `neutral`, `negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finclf_bert_target_sentiments_es_1.0.0_3.0_1679657417647.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finclf_bert_target_sentiments_es_1.0.0_3.0_1679657417647.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
sequenceClassifier = finance.BertForSequenceClassification.pretrained("finclf_bert_target_sentiments","es","finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("class")\
  .setCaseSensitive(True)

pipeline =  nlp.Pipeline(
    stages=[
  documentAssembler,
  tokenizer,
  sequenceClassifier
    ]
)

```

</div>

## Results

```bash
+-------------------------------------------------------------------------------------------+----------+
|text                                                                                       |result    |
+-------------------------------------------------------------------------------------------+----------+
|La deuda de las familias cae en 25.000 millones en 2015 y marca niveles previos a la crisis|[positive]|
+-------------------------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_target_sentiments|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|408.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://codalab.lisn.upsaclay.fr/competitions/10052#learn_the_details

## Benchmarking

```bash
 
labels      precision    recall  f1-score   support
negative        0.81      0.66      0.73        44
neutral         1.00      0.22      0.36         9
positive        0.82      0.93      0.87       105
accuracy          -        -        0.82       158
macro-avg       0.87      0.60      0.65       158
weighted-avg    0.82      0.82      0.80       158

```