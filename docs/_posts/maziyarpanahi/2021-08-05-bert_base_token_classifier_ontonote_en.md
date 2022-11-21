---
layout: model
title: BERT Token Classification - NER OntoNotes (bert_base_token_classifier_ontonote)
author: John Snow Labs
name: bert_base_token_classifier_ontonote
date: 2021-08-05
tags: [ner, en, english, token_classification, bert, open_source, ontonotes]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`BERT Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.


**bert_base_token_classifier_ontonote** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, and WORK_OF_ART.

We used [TFBertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html#tfbertfortokenclassification) to train this model and used `BertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_token_classifier_ontonote_en_3.2.0_2.4_1628174984240.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification \
.pretrained('bert_base_token_classifier_ontonote', 'en') \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_base_token_classifier_ontonote", "en")
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
nlu.load("en.classify.token_bert.ontonote").predict("""My name is John!""")
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
|Model Name:|bert_base_token_classifier_ontonote|
|Compatibility:|Spark NLP 3.2.0+|
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
Test:

precision    recall  f1-score   support

B-CARDINAL       0.86      0.88      0.87       935
B-DATE       0.88      0.90      0.89      1602
B-EVENT       0.71      0.62      0.66        63
B-FAC       0.75      0.76      0.76       135
B-GPE       0.97      0.91      0.94      2240
B-LANGUAGE       0.79      0.68      0.73        22
B-LAW       0.76      0.65      0.70        40
B-LOC       0.78      0.83      0.80       179
B-MONEY       0.88      0.90      0.89       314
B-NORP       0.92      0.96      0.94       841
B-ORDINAL       0.81      0.93      0.87       195
B-ORG       0.87      0.89      0.88      1795
B-PERCENT       0.92      0.95      0.93       349
B-PERSON       0.96      0.95      0.95      1988
B-PRODUCT       0.75      0.78      0.76        76
B-QUANTITY       0.81      0.82      0.82       105
B-TIME       0.69      0.70      0.69       212
B-WORK_OF_ART       0.66      0.74      0.70       166
I-CARDINAL       0.83      0.88      0.86       331
I-DATE       0.89      0.92      0.90      2011
I-EVENT       0.70      0.71      0.70       130
I-FAC       0.82      0.85      0.84       213
I-GPE       0.96      0.89      0.93       628
I-LAW       0.85      0.65      0.74       106
I-LOC       0.85      0.84      0.84       180
I-MONEY       0.94      0.96      0.95       685
I-NORP       0.99      0.79      0.88       160
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.91      0.93      0.92      2406
I-PERCENT       0.95      0.95      0.95       523
I-PERSON       0.96      0.96      0.96      1412
I-PRODUCT       0.81      0.80      0.80        69
I-QUANTITY       0.83      0.92      0.87       206
I-TIME       0.71      0.77      0.74       255
I-WORK_OF_ART       0.71      0.66      0.68       337
O       0.99      0.99      0.99    131815

accuracy                           0.98    152728
macro avg       0.82      0.81      0.82    152728
weighted avg       0.98      0.98      0.98    152728



processed 152728 tokens with 11257 phrases; found: 11537 phrases; correct: 9906.
accuracy:  90.22%; (non-O)
accuracy:  98.00%; precision:  85.86%; recall:  88.00%; FB1:  86.92
CARDINAL: precision:  83.35%; recall:  86.20%; FB1:  84.75  967
DATE: precision:  82.23%; recall:  86.95%; FB1:  84.53  1694
EVENT: precision:  58.06%; recall:  57.14%; FB1:  57.60  62
FAC: precision:  68.67%; recall:  76.30%; FB1:  72.28  150
GPE: precision:  95.59%; recall:  90.00%; FB1:  92.71  2109
LANGUAGE: precision:  78.95%; recall:  68.18%; FB1:  73.17  19
LAW: precision:  63.16%; recall:  60.00%; FB1:  61.54  38
LOC: precision:  71.29%; recall:  80.45%; FB1:  75.59  202
MONEY: precision:  85.40%; recall:  87.58%; FB1:  86.48  322
NORP: precision:  89.82%; recall:  93.34%; FB1:  91.55  874
ORDINAL: precision:  81.17%; recall:  92.82%; FB1:  86.60  223
ORG: precision:  82.80%; recall:  86.35%; FB1:  84.54  1872
PERCENT: precision:  86.23%; recall:  89.68%; FB1:  87.92  363
PERSON: precision:  93.93%; recall:  94.22%; FB1:  94.07  1994
PRODUCT: precision:  70.37%; recall:  75.00%; FB1:  72.61  81
QUANTITY: precision:  73.28%; recall:  80.95%; FB1:  76.92  116
TIME: precision:  58.02%; recall:  66.51%; FB1:  61.98  243
WORK_OF_ART: precision:  52.40%; recall:  65.66%; FB1:  58.29  208
```