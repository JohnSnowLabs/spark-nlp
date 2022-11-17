---
layout: model
title: RoBERTa NER (Base, OntoNotes)
author: John Snow Labs
name: roberta_base_token_classifier_ontonotes
date: 2021-09-26
tags: [ner, ontonotes, open_source, token_classification, en, english, roberta]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
recommended: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`RoBERTa Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.


**roberta_base_token_classifier_ontonotes** is a fine-tuned RoBERTa model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, and WORK_OF_ART.

We used [TFRobertaForTokenClassification](https://huggingface.co/transformers/model_doc/roberta.html#tfrobertafortokenclassification) to train this model and used `RoBertaForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_token_classifier_ontonotes_en_3.3.0_3.0_1632675229237.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification \
.pretrained('roberta_base_token_classifier_ontonotes', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('ner') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
tokenClassifier,
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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_base_token_classifier_ontonotes", "en")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_roberta_base_token_classifier_ontonotes").predict("""My name is John!""")
```

</div>

## Results

```bash
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PERSON, I-PERSON, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PERSON, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_token_classifier_ontonotes|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://catalog.ldc.upenn.edu/LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)

## Benchmarking

```bash
precision    recall  f1-score   support

B-CARDINAL       0.83      0.91      0.87       938
B-DATE       0.88      0.85      0.87      1507
B-EVENT       0.79      0.64      0.71       143
B-FAC       0.64      0.66      0.65       115
B-GPE       0.95      0.89      0.92      2268
B-LANGUAGE       0.96      0.73      0.83        33
B-LAW       0.77      0.75      0.76        40
B-LOC       0.72      0.77      0.75       204
B-MONEY       0.94      0.93      0.94       274
B-NORP       0.88      0.92      0.90       847
B-ORDINAL       0.83      0.81      0.82       232
B-ORG       0.87      0.88      0.88      1740
B-PERCENT       0.91      0.92      0.91       177
B-PERSON       0.92      0.95      0.94      2020
B-PRODUCT       0.74      0.74      0.74        72
B-QUANTITY       0.86      0.85      0.85       100
B-TIME       0.79      0.73      0.76       214
B-WORK_OF_ART       0.65      0.63      0.64       142
I-CARDINAL       0.84      0.88      0.86       290
I-DATE       0.89      0.89      0.89      1809
I-EVENT       0.82      0.70      0.76       272
I-FAC       0.69      0.74      0.71       203
I-GPE       0.91      0.85      0.87       555
I-LANGUAGE       0.00      0.00      0.00         0
I-LAW       0.75      0.68      0.71        84
I-LOC       0.69      0.76      0.72       188
I-MONEY       0.96      0.99      0.97       587
I-NORP       0.78      0.66      0.72        44
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.91      0.91      0.91      2336
I-PERCENT       0.90      0.97      0.94       258
I-PERSON       0.94      0.97      0.96      1395
I-PRODUCT       0.88      0.88      0.88       129
I-QUANTITY       0.88      0.92      0.90       209
I-TIME       0.75      0.74      0.75       260
I-WORK_OF_ART       0.65      0.75      0.70       334
O       0.99      0.99      0.99    127701

accuracy                           0.98    147724
macro avg       0.79      0.78      0.78    147724
weighted avg       0.98      0.98      0.98    147724



processed 147724 tokens with 11066 phrases; found: 11196 phrases; correct: 9582.
accuracy:  88.60%; (non-O)
accuracy:  97.79%; precision:  85.58%; recall:  86.59%; FB1:  86.08
CARDINAL: precision:  81.30%; recall:  89.45%; FB1:  85.18  1032
DATE: precision:  84.09%; recall:  83.15%; FB1:  83.62  1490
EVENT: precision:  71.19%; recall:  58.74%; FB1:  64.37  118
FAC: precision:  59.68%; recall:  64.35%; FB1:  61.92  124
GPE: precision:  93.60%; recall:  88.32%; FB1:  90.88  2140
LANGUAGE: precision:  92.00%; recall:  69.70%; FB1:  79.31  25
LAW: precision:  67.50%; recall:  67.50%; FB1:  67.50  40
LOC: precision:  66.23%; recall:  75.00%; FB1:  70.34  231
MONEY: precision:  93.12%; recall:  93.80%; FB1:  93.45  276
NORP: precision:  86.77%; recall:  91.38%; FB1:  89.02  892
ORDINAL: precision:  83.26%; recall:  81.47%; FB1:  82.35  227
ORG: precision:  82.75%; recall:  84.89%; FB1:  83.80  1785
PERCENT: precision:  88.95%; recall:  90.96%; FB1:  89.94  181
PERSON: precision:  90.97%; recall:  94.31%; FB1:  92.61  2094
PRODUCT: precision:  71.05%; recall:  75.00%; FB1:  72.97  76
QUANTITY: precision:  75.47%; recall:  80.00%; FB1:  77.67  106
TIME: precision:  70.48%; recall:  69.16%; FB1:  69.81  210
WORK_OF_ART: precision:  54.36%; recall:  57.04%; FB1:  55.67  149
```
