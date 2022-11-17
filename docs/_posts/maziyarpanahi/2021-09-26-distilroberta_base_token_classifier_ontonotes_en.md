---
layout: model
title: DistilRoBERTa Token Classification - NER OntoNotes (distilroberta_base_token_classifier_ontonotes)
author: John Snow Labs
name: distilroberta_base_token_classifier_ontonotes
date: 2021-09-26
tags: [roberta, en, english, ontonotes, open_source, ner, token_classification]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`RoBERTa Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.


**distilroberta_base_token_classifier_ontonotes** is a fine-tuned RoBERTa model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, and WORK_OF_ART.

We used [TFRobertaForTokenClassification](https://huggingface.co/transformers/model_doc/roberta.html#tfrobertafortokenclassification) to train this model and used `RoBertaForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilroberta_base_token_classifier_ontonotes_en_3.3.0_3.0_1632673971874.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('distilroberta_base_token_classifier_ontonotes', 'en') \
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

val tokenClassifier = RoBertaForTokenClassification.pretrained("distilroberta_base_token_classifier_ontonotes", "en")
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
nlu.load("en.classify.token_distilroberta_base_token_classifier_ontonotes").predict("""My name is John!""")
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
|Model Name:|distilroberta_base_token_classifier_ontonotes|
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
B-CARDINAL       0.82      0.86      0.84       935
B-DATE       0.86      0.85      0.85      1602
B-EVENT       0.69      0.56      0.61        63
B-FAC       0.66      0.59      0.62       135
B-GPE       0.94      0.89      0.91      2240
B-LANGUAGE       0.91      0.45      0.61        22
B-LAW       0.91      0.53      0.67        40
B-LOC       0.69      0.71      0.70       179
B-MONEY       0.86      0.89      0.87       314
B-NORP       0.84      0.87      0.85       841
B-ORDINAL       0.81      0.89      0.85       195
B-ORG       0.85      0.83      0.84      1795
B-PERCENT       0.92      0.92      0.92       349
B-PERSON       0.92      0.93      0.93      1988
B-PRODUCT       0.64      0.64      0.64        76
B-QUANTITY       0.73      0.76      0.74       105
B-TIME       0.71      0.54      0.61       212
B-WORK_OF_ART       0.72      0.52      0.61       166
I-CARDINAL       0.82      0.77      0.80       331
I-DATE       0.87      0.88      0.88      2011
I-EVENT       0.69      0.79      0.74       130
I-FAC       0.72      0.73      0.73       213
I-GPE       0.90      0.77      0.83       628
I-LAW       0.98      0.60      0.75       106
I-LOC       0.79      0.68      0.73       180
I-MONEY       0.92      0.96      0.94       685
I-NORP       0.86      0.57      0.68       160
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.89      0.92      0.90      2406
I-PERCENT       0.93      0.96      0.94       523
I-PERSON       0.94      0.92      0.93      1412
I-PRODUCT       0.69      0.71      0.70        69
I-QUANTITY       0.79      0.87      0.82       206
I-TIME       0.73      0.78      0.75       255
I-WORK_OF_ART       0.70      0.57      0.63       337
O       0.99      0.99      0.99    131815

accuracy                           0.97    152728
macro avg       0.80      0.74      0.76    152728
weighted avg       0.97      0.97      0.97    152728



processed 152728 tokens with 11257 phrases; found: 11382 phrases; correct: 9305.
accuracy:  85.75%; (non-O)
accuracy:  97.36%; precision:  81.75%; recall:  82.66%; FB1:  82.20
CARDINAL: precision:  79.09%; recall:  83.74%; FB1:  81.35  990
DATE: precision:  78.48%; recall:  81.02%; FB1:  79.73  1654
EVENT: precision:  57.14%; recall:  57.14%; FB1:  57.14  63
FAC: precision:  58.52%; recall:  58.52%; FB1:  58.52  135
GPE: precision:  90.96%; recall:  86.21%; FB1:  88.52  2123
LANGUAGE: precision:  90.91%; recall:  45.45%; FB1:  60.61  11
LAW: precision:  68.97%; recall:  50.00%; FB1:  57.97  29
LOC: precision:  63.30%; recall:  66.48%; FB1:  64.85  188
MONEY: precision:  78.90%; recall:  86.94%; FB1:  82.73  346
NORP: precision:  82.29%; recall:  86.21%; FB1:  84.20  881
ORDINAL: precision:  80.84%; recall:  88.72%; FB1:  84.60  214
ORG: precision:  78.98%; recall:  79.55%; FB1:  79.27  1808
PERCENT: precision:  85.99%; recall:  87.97%; FB1:  86.97  357
PERSON: precision:  88.43%; recall:  90.74%; FB1:  89.57  2040
PRODUCT: precision:  58.97%; recall:  60.53%; FB1:  59.74  78
QUANTITY: precision:  57.14%; recall:  72.38%; FB1:  63.87  133
TIME: precision:  59.44%; recall:  50.47%; FB1:  54.59  180
WORK_OF_ART: precision:  59.21%; recall:  54.22%; FB1:  56.60  152

```