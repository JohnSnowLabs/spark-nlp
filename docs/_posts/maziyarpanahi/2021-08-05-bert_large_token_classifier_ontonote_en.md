---
layout: model
title: BERT Token Classification Large - NER OntoNotes (bert_large_token_classifier_ontonote)
author: John Snow Labs
name: bert_large_token_classifier_ontonote
date: 2021-08-05
tags: [open_source, ner, en, english, bert, token_classification, ontonotes, large]
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


**bert_large_token_classifier_ontonote** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, and WORK_OF_ART.

We used [TFBertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html#tfbertfortokenclassification) to train this model and used `BertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_token_classifier_ontonote_en_3.2.0_2.4_1628176479421.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_large_token_classifier_ontonote_en_3.2.0_2.4_1628176479421.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
.pretrained('bert_large_token_classifier_ontonote', 'en') \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_large_token_classifier_ontonote", "en")
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
nlu.load("en.classify.token_bert.large_ontonote").predict("""My name is John!""")
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
|Model Name:|bert_large_token_classifier_ontonote|
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

B-CARDINAL       0.86      0.86      0.86       935
B-DATE       0.88      0.89      0.88      1602
B-EVENT       0.76      0.67      0.71        63
B-FAC       0.77      0.84      0.81       135
B-GPE       0.98      0.92      0.95      2240
B-LANGUAGE       0.79      0.68      0.73        22
B-LAW       0.77      0.68      0.72        40
B-LOC       0.73      0.82      0.78       179
B-MONEY       0.90      0.89      0.89       314
B-NORP       0.94      0.96      0.95       841
B-ORDINAL       0.82      0.91      0.87       195
B-ORG       0.90      0.91      0.91      1795
B-PERCENT       0.94      0.93      0.94       349
B-PERSON       0.95      0.96      0.95      1988
B-PRODUCT       0.79      0.80      0.80        76
B-QUANTITY       0.82      0.83      0.82       105
B-TIME       0.69      0.69      0.69       212
B-WORK_OF_ART       0.71      0.72      0.71       166
I-CARDINAL       0.83      0.89      0.86       331
I-DATE       0.90      0.90      0.90      2011
I-EVENT       0.76      0.74      0.75       130
I-FAC       0.79      0.91      0.85       213
I-GPE       0.94      0.89      0.92       628
I-LAW       0.82      0.66      0.73       106
I-LOC       0.89      0.83      0.86       180
I-MONEY       0.94      0.96      0.95       685
I-NORP       0.98      0.91      0.94       160
I-ORDINAL       0.00      0.00      0.00         4
I-ORG       0.92      0.93      0.93      2406
I-PERCENT       0.96      0.95      0.96       523
I-PERSON       0.97      0.94      0.96      1412
I-PRODUCT       0.81      0.81      0.81        69
I-QUANTITY       0.87      0.92      0.89       206
I-TIME       0.68      0.73      0.70       255
I-WORK_OF_ART       0.72      0.66      0.69       337
O       0.99      0.99      0.99    131815

accuracy                           0.98    152728
macro avg       0.83      0.82      0.82    152728
weighted avg       0.98      0.98      0.98    152728



processed 152728 tokens with 11257 phrases; found: 11394 phrases; correct: 10001.
accuracy:  90.30%; (non-O)
accuracy:  98.10%; precision:  87.77%; recall:  88.84%; FB1:  88.31
CARDINAL: precision:  83.37%; recall:  84.17%; FB1:  83.77  944
DATE: precision:  83.84%; recall:  86.14%; FB1:  84.98  1646
EVENT: precision:  64.06%; recall:  65.08%; FB1:  64.57  64
FAC: precision:  69.38%; recall:  82.22%; FB1:  75.25  160
GPE: precision:  96.64%; recall:  91.25%; FB1:  93.87  2115
LANGUAGE: precision:  78.95%; recall:  68.18%; FB1:  73.17  19
LAW: precision:  54.76%; recall:  57.50%; FB1:  56.10  42
LOC: precision:  70.10%; recall:  79.89%; FB1:  74.67  204
MONEY: precision:  87.70%; recall:  88.54%; FB1:  88.11  317
NORP: precision:  93.60%; recall:  95.72%; FB1:  94.65  860
ORDINAL: precision:  82.41%; recall:  91.28%; FB1:  86.62  216
ORG: precision:  87.26%; recall:  89.30%; FB1:  88.27  1837
PERCENT: precision:  89.43%; recall:  89.68%; FB1:  89.56  350
PERSON: precision:  93.70%; recall:  95.02%; FB1:  94.36  2016
PRODUCT: precision:  68.29%; recall:  73.68%; FB1:  70.89  82
QUANTITY: precision:  78.57%; recall:  83.81%; FB1:  81.11  112
TIME: precision:  58.85%; recall:  62.74%; FB1:  60.73  226
WORK_OF_ART: precision:  61.96%; recall:  68.67%; FB1:  65.14  184
```