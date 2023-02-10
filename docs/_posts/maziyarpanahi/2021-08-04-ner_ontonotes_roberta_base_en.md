---
layout: model
title: Named Entity Recognition - OntoNotes RoBERTa (ner_ontonotes_roberta_base)
author: John Snow Labs
name: ner_ontonotes_roberta_base
date: 2021-08-04
tags: [ner, roberta, open_source, english, en, ontonotes]
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

`ner_ontonotes_roberta_base` is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `roberta_base` model from the `RoBertaEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ontonotes_roberta_base_en_3.2.0_2.4_1628078208687.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_ontonotes_roberta_base_en_3.2.0_2.4_1628078208687.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
.pretrained('roberta_base', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_ontonotes_roberta_base', 'en') \
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

val embeddings = RoBertaEmbeddings.pretrained("roberta_base", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_ontonotes_roberta_base", "en") 
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

ner_df = nlu.load('en.ner.ner_ontonotes_roberta_base').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ontonotes_roberta_base|
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

B-CARDINAL       0.85      0.87      0.86       935
B-DATE       0.86      0.89      0.88      1602
B-EVENT       0.65      0.51      0.57        63
B-FAC       0.79      0.56      0.66       135
B-GPE       0.97      0.92      0.94      2240
B-LANGUAGE       0.82      0.41      0.55        22
B-LAW       0.59      0.57      0.58        40
B-LOC       0.81      0.74      0.77       179
B-MONEY       0.89      0.91      0.90       314
B-NORP       0.92      0.95      0.93       841
B-ORDINAL       0.81      0.88      0.84       195
B-ORG       0.86      0.92      0.89      1795
B-PERCENT       0.93      0.92      0.93       349
B-PERSON       0.93      0.93      0.93      1988
B-PRODUCT       0.72      0.63      0.67        76
B-QUANTITY       0.81      0.81      0.81       105
B-TIME       0.64      0.67      0.65       212
B-WORK_OF_ART       0.72      0.57      0.64       166
I-CARDINAL       0.84      0.81      0.82       331
I-DATE       0.87      0.92      0.90      2011
I-EVENT       0.71      0.66      0.69       130
I-FAC       0.79      0.67      0.72       213
I-GPE       0.91      0.89      0.90       628
I-LAW       0.77      0.64      0.70       106
I-LOC       0.86      0.71      0.78       180
I-MONEY       0.94      0.97      0.95       685
I-NORP       0.91      0.85      0.88       160
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.89      0.94      0.91      2406
I-PERCENT       0.95      0.96      0.96       523
I-PERSON       0.94      0.93      0.94      1412
I-PRODUCT       0.70      0.70      0.70        69
I-QUANTITY       0.89      0.90      0.89       206
I-TIME       0.64      0.74      0.68       255
I-WORK_OF_ART       0.75      0.55      0.64       337
O       0.99      0.99      0.99    131815

accuracy                           0.98    152728
macro avg       0.80      0.76      0.78    152728
weighted avg       0.98      0.98      0.98    152728


processed 152728 tokens with 11257 phrases; found: 11277 phrases; correct: 9868.
accuracy:  89.00%; (non-O)
accuracy:  97.82%; precision:  87.51%; recall:  87.66%; FB1:  87.58
CARDINAL: precision:  84.05%; recall:  86.20%; FB1:  85.11  959
DATE: precision:  84.18%; recall:  87.02%; FB1:  85.57  1656
EVENT: precision:  63.27%; recall:  49.21%; FB1:  55.36  49
FAC: precision:  77.08%; recall:  54.81%; FB1:  64.07  96
GPE: precision:  96.03%; recall:  91.74%; FB1:  93.84  2140
LANGUAGE: precision:  81.82%; recall:  40.91%; FB1:  54.55  11
LAW: precision:  56.41%; recall:  55.00%; FB1:  55.70  39
LOC: precision:  79.63%; recall:  72.07%; FB1:  75.66  162
MONEY: precision:  87.81%; recall:  89.49%; FB1:  88.64  320
NORP: precision:  91.10%; recall:  93.70%; FB1:  92.38  865
ORDINAL: precision:  80.66%; recall:  87.69%; FB1:  84.03  212
ORG: precision:  84.05%; recall:  89.25%; FB1:  86.57  1906
PERCENT: precision:  90.20%; recall:  89.68%; FB1:  89.94  347
PERSON: precision:  92.66%; recall:  92.66%; FB1:  92.66  1988
PRODUCT: precision:  68.66%; recall:  60.53%; FB1:  64.34  67
QUANTITY: precision:  80.95%; recall:  80.95%; FB1:  80.95  105
TIME: precision:  60.00%; recall:  63.68%; FB1:  61.78  225
WORK_OF_ART: precision:  65.38%; recall:  51.20%; FB1:  57.43  130
```
