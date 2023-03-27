---
layout: model
title: Financial Company Sentiments (Codalab)
author: John Snow Labs
name: finclf_bert_company_sentiments
date: 2023-03-27
tags: [bert, finance, classification, es, licensed, tensorflow]
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

This  Spanish Text Classifier will identify from the viewpoint of a company whether a financial statement is `positive`, `neutral` or `negative`. This model is trained from the competition - `IBERLEF 2023 Task - FinancES. Financial Targeted Sentiment Analysis in Spanish`. We have used the participation dataset which is a small subset of the main one to train this model.

## Predicted Entities

`positive`, `neutral`, `negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_company_sentiments_es_1.0.0_3.0_1679941143258.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_bert_company_sentiments_es_1.0.0_3.0_1679941143258.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
sequenceClassifier = finance.BertForSequenceClassification.pretrained("finclf_bert_company_sentiments","es","finance/models")\
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
+-----------------------------------------------------------------------------+---------+
|text                                                                         |result   |
+-----------------------------------------------------------------------------+---------+
|Leopoldo del Nogal abandona el consejo de administraciÃ³n de El Corte InglÃ©s|[neutral]|
+-----------------------------------------------------------------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_company_sentiments|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
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
 
labels            precision    recall  f1-score   support
    negative       0.51      0.52      0.52        44
     neutral       0.65      0.74      0.69        77
    positive       0.56      0.38      0.45        37
    accuracy        -         -        0.59       158
   macro avg       0.57      0.55      0.55       158
weighted avg       0.59      0.59      0.59       158

```