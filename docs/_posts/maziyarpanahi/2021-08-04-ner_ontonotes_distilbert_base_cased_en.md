---
layout: model
title: Named Entity Recognition - OntoNotes DistilBERT (ner_ontonotes_distilbert_base_cased)
author: John Snow Labs
name: ner_ontonotes_distilbert_base_cased
date: 2021-08-04
tags: [ner, en, english, distilbert, ontonotes, open_source]
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

`ner_ontonotes_distilbert_base_cased` is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `distilbert_base_cased` model from the `DistilBertEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ontonotes_distilbert_base_cased_en_3.2.0_2.4_1628079072311.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = DistilBertEmbeddings\
.pretrained('distilbert_base_cased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_ontonotes_distilbert_base_cased', 'en') \
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

val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_ontonotes_distilbert_base_cased", "en") 
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

ner_df = nlu.load('en.ner.ner_ontonotes_distilbert_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ontonotes_distilbert_base_cased|
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

B-CARDINAL       0.85      0.85      0.85       935
B-DATE       0.87      0.87      0.87      1602
B-EVENT       0.60      0.56      0.58        63
B-FAC       0.69      0.69      0.69       135
B-GPE       0.96      0.93      0.95      2240
B-LANGUAGE       0.90      0.41      0.56        22
B-LAW       0.74      0.42      0.54        40
B-LOC       0.68      0.80      0.74       179
B-MONEY       0.90      0.92      0.91       314
B-NORP       0.94      0.94      0.94       841
B-ORDINAL       0.84      0.87      0.85       195
B-ORG       0.88      0.89      0.88      1795
B-PERCENT       0.92      0.92      0.92       349
B-PERSON       0.93      0.93      0.93      1988
B-PRODUCT       0.60      0.68      0.64        76
B-QUANTITY       0.80      0.74      0.77       105
B-TIME       0.70      0.57      0.62       212
B-WORK_OF_ART       0.77      0.58      0.66       166
I-CARDINAL       0.77      0.90      0.83       331
I-DATE       0.87      0.92      0.89      2011
I-EVENT       0.61      0.78      0.68       130
I-FAC       0.76      0.81      0.79       213
I-GPE       0.95      0.86      0.90       628
I-LAW       0.90      0.54      0.67       106
I-LOC       0.72      0.80      0.76       180
I-MONEY       0.94      0.98      0.96       685
I-NORP       0.96      0.85      0.90       160
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.89      0.92      0.91      2406
I-PERCENT       0.95      0.95      0.95       523
I-PERSON       0.95      0.94      0.94      1412
I-PRODUCT       0.59      0.81      0.68        69
I-QUANTITY       0.88      0.83      0.85       206
I-TIME       0.72      0.65      0.68       255
I-WORK_OF_ART       0.81      0.57      0.67       337
O       0.99      0.99      0.99    131815

accuracy                           0.98    152728
macro avg       0.80      0.77      0.78    152728
weighted avg       0.98      0.98      0.98    152728


processed 152728 tokens with 11257 phrases; found: 11127 phrases; correct: 9747.
accuracy:  88.49%; (non-O)
accuracy:  97.78%; precision:  87.60%; recall:  86.59%; FB1:  87.09
CARDINAL: precision:  83.58%; recall:  83.85%; FB1:  83.72  938
DATE: precision:  84.94%; recall:  84.52%; FB1:  84.73  1594
EVENT: precision:  58.62%; recall:  53.97%; FB1:  56.20  58
FAC: precision:  68.66%; recall:  68.15%; FB1:  68.40  134
GPE: precision:  95.96%; recall:  92.37%; FB1:  94.13  2156
LANGUAGE: precision:  90.00%; recall:  40.91%; FB1:  56.25  10
LAW: precision:  69.57%; recall:  40.00%; FB1:  50.79  23
LOC: precision:  65.40%; recall:  77.09%; FB1:  70.77  211
MONEY: precision:  88.79%; recall:  90.76%; FB1:  89.76  321
NORP: precision:  93.45%; recall:  93.34%; FB1:  93.40  840
ORDINAL: precision:  83.74%; recall:  87.18%; FB1:  85.43  203
ORG: precision:  85.34%; recall:  86.57%; FB1:  85.95  1821
PERCENT: precision:  89.02%; recall:  88.25%; FB1:  88.63  346
PERSON: precision:  91.40%; recall:  91.45%; FB1:  91.43  1989
PRODUCT: precision:  58.14%; recall:  65.79%; FB1:  61.73  86
QUANTITY: precision:  78.79%; recall:  74.29%; FB1:  76.47  99
TIME: precision:  65.70%; recall:  53.30%; FB1:  58.85  172
WORK_OF_ART: precision:  71.43%; recall:  54.22%; FB1:  61.64  126
```