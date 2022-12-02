---
layout: model
title: Named Entity Recognition - OntoNotes RoBERTa (ner_ontonotes_roberta_large)
author: John Snow Labs
name: ner_ontonotes_roberta_large
date: 2021-08-04
tags: [en, english, open_source, roberta, ner, ontonotes]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`ner_ontonotes_roberta_large` is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `roberta_large` model from the `RoBertaEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ontonotes_roberta_large_en_3.2.0_2.4_1628078836777.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = RoBertaEmbeddings\
.pretrained('roberta_large', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_ontonotes_roberta_large', 'en') \
.setInputCols(['document', 'token', 'embeddings']) \
.setOutputCol('ner')

ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
embeddings,
ner_model,
ner_converter
])

example = spark.createDataFrame([['My name is John!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_large", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_ontonotes_roberta_large", "en") 
.setInputCols("document"', "token", "embeddings") 
.setOutputCol("ner")

val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner_model, ner_converter))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```

{:.nlu-block}
```python
import nlu

text = ["My name is John!"]

ner_df = nlu.load('en.ner.ner_ontonotes_roberta_large').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ontonotes_roberta_large|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

[https://catalog.ldc.upenn.edu/LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)

## Benchmarking

```bash
precision    recall  f1-score   support

B-CARDINAL       0.86      0.84      0.85       935
B-DATE       0.86      0.90      0.88      1602
B-EVENT       0.68      0.67      0.67        63
B-FAC       0.67      0.68      0.67       135
B-GPE       0.96      0.96      0.96      2240
B-LANGUAGE       0.92      0.50      0.65        22
B-LAW       0.68      0.62      0.65        40
B-LOC       0.88      0.75      0.81       179
B-MONEY       0.88      0.91      0.90       314
B-NORP       0.93      0.93      0.93       841
B-ORDINAL       0.82      0.90      0.86       195
B-ORG       0.90      0.91      0.91      1795
B-PERCENT       0.94      0.92      0.93       349
B-PERSON       0.95      0.94      0.94      1988
B-PRODUCT       0.83      0.63      0.72        76
B-QUANTITY       0.77      0.80      0.79       105
B-TIME       0.68      0.71      0.69       212
B-WORK_OF_ART       0.60      0.62      0.61       166
I-CARDINAL       0.81      0.81      0.81       331
I-DATE       0.86      0.94      0.90      2011
I-EVENT       0.78      0.81      0.80       130
I-FAC       0.68      0.83      0.74       213
I-GPE       0.94      0.89      0.91       628
I-LAW       0.87      0.64      0.74       106
I-LOC       0.93      0.71      0.81       180
I-MONEY       0.92      0.97      0.95       685
I-NORP       0.98      0.72      0.83       160
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.91      0.94      0.92      2406
I-PERCENT       0.95      0.95      0.95       523
I-PERSON       0.96      0.94      0.95      1412
I-PRODUCT       0.89      0.74      0.81        69
I-QUANTITY       0.84      0.90      0.87       206
I-TIME       0.68      0.75      0.72       255
I-WORK_OF_ART       0.66      0.64      0.65       337
O       0.99      0.99      0.99    131815

accuracy                           0.98    152728
macro avg       0.82      0.79      0.80    152728
weighted avg       0.98      0.98      0.98    152728


processed 152728 tokens with 11257 phrases; found: 11320 phrases; correct: 9995.
accuracy:  90.20%; (non-O)
accuracy:  97.94%; precision:  88.30%; recall:  88.79%; FB1:  88.54
CARDINAL: precision:  84.54%; recall:  82.46%; FB1:  83.49  912
DATE: precision:  84.12%; recall:  87.95%; FB1:  85.99  1675
EVENT: precision:  66.13%; recall:  65.08%; FB1:  65.60  62
FAC: precision:  65.94%; recall:  67.41%; FB1:  66.67  138
GPE: precision:  95.70%; recall:  95.40%; FB1:  95.55  2233
LANGUAGE: precision:  91.67%; recall:  50.00%; FB1:  64.71  12
LAW: precision:  64.86%; recall:  60.00%; FB1:  62.34  37
LOC: precision:  86.84%; recall:  73.74%; FB1:  79.76  152
MONEY: precision:  87.38%; recall:  90.45%; FB1:  88.89  325
NORP: precision:  92.18%; recall:  92.51%; FB1:  92.34  844
ORDINAL: precision:  81.86%; recall:  90.26%; FB1:  85.85  215
ORG: precision:  87.58%; recall:  89.19%; FB1:  88.38  1828
PERCENT: precision:  90.67%; recall:  89.11%; FB1:  89.88  343
PERSON: precision:  93.90%; recall:  93.61%; FB1:  93.75  1982
PRODUCT: precision:  82.76%; recall:  63.16%; FB1:  71.64  58
QUANTITY: precision:  76.15%; recall:  79.05%; FB1:  77.57  109
TIME: precision:  63.68%; recall:  66.98%; FB1:  65.29  223
WORK_OF_ART: precision:  55.23%; recall:  57.23%; FB1:  56.21  172
```
