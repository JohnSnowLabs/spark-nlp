---
layout: model
title: DistilBERT Token Classification - NER OntoNotes (distilbert_base_token_classifier_ontonotes)
author: John Snow Labs
name: distilbert_base_token_classifier_ontonotes
date: 2021-08-05
tags: [ner, ontonotes, distilbert, token_classification, en, english, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`DistilBERT Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.


**distilbert_base_token_classifier_ontonotes** is a fine-tuned DistilBERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, and WORK_OF_ART.

We used [TFDistilBertForTokenClassification](https://huggingface.co/transformers/model_doc/distilbert.html#tfdistilbertfortokenclassification) to train this model and used `DistilBertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_token_classifier_ontonotes_en_3.2.0_2.4_1628181511882.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = DistilBertForTokenClassification \
      .pretrained('distilbert_base_token_classifier_ontonotes', 'en') \
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

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_base_token_classifier_ontonotes", "en")
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
|Model Name:|distilbert_base_token_classifier_ontonotes|
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

   B-CARDINAL       0.85      0.86      0.86       935
       B-DATE       0.87      0.90      0.88      1602
      B-EVENT       0.73      0.59      0.65        63
        B-FAC       0.72      0.71      0.71       135
        B-GPE       0.96      0.91      0.94      2240
   B-LANGUAGE       0.92      0.55      0.69        22
        B-LAW       0.80      0.60      0.69        40
        B-LOC       0.78      0.78      0.78       179
      B-MONEY       0.89      0.89      0.89       314
       B-NORP       0.90      0.93      0.91       841
    B-ORDINAL       0.81      0.90      0.86       195
        B-ORG       0.87      0.89      0.88      1795
    B-PERCENT       0.94      0.93      0.93       349
     B-PERSON       0.93      0.94      0.94      1988
    B-PRODUCT       0.59      0.74      0.65        76
   B-QUANTITY       0.80      0.78      0.79       105
       B-TIME       0.71      0.64      0.67       212
B-WORK_OF_ART       0.63      0.67      0.65       166
   I-CARDINAL       0.80      0.91      0.85       331
       I-DATE       0.88      0.91      0.90      2011
      I-EVENT       0.71      0.71      0.71       130
        I-FAC       0.80      0.86      0.83       213
        I-GPE       0.92      0.90      0.91       628
        I-LAW       0.93      0.61      0.74       106
        I-LOC       0.83      0.81      0.82       180
      I-MONEY       0.93      0.96      0.95       685
       I-NORP       0.95      0.76      0.84       160
    I-ORDINAL       0.00      0.00      0.00         4
        I-ORG       0.89      0.93      0.91      2406
    I-PERCENT       0.93      0.97      0.95       523
     I-PERSON       0.95      0.93      0.94      1412
    I-PRODUCT       0.64      0.78      0.70        69
   I-QUANTITY       0.80      0.88      0.84       206
       I-TIME       0.71      0.79      0.75       255
I-WORK_OF_ART       0.69      0.63      0.65       337
            O       0.99      0.99      0.99    131815

     accuracy                           0.98    152728
    macro avg       0.81      0.79      0.80    152728
 weighted avg       0.98      0.98      0.98    152728



processed 152728 tokens with 11257 phrases; found: 11586 phrases; correct: 9747.
accuracy:  89.24%; (non-O)
accuracy:  97.80%; precision:  84.13%; recall:  86.59%; FB1:  85.34
         CARDINAL: precision:  82.90%; recall:  85.03%; FB1:  83.95  959
             DATE: precision:  80.79%; recall:  86.64%; FB1:  83.61  1718
            EVENT: precision:  57.63%; recall:  53.97%; FB1:  55.74  59
              FAC: precision:  63.01%; recall:  68.15%; FB1:  65.48  146
              GPE: precision:  93.96%; recall:  90.31%; FB1:  92.10  2153
         LANGUAGE: precision:  92.31%; recall:  54.55%; FB1:  68.57  13
              LAW: precision:  55.56%; recall:  50.00%; FB1:  52.63  36
              LOC: precision:  71.35%; recall:  76.54%; FB1:  73.85  192
            MONEY: precision:  84.00%; recall:  86.94%; FB1:  85.45  325
             NORP: precision:  87.96%; recall:  91.20%; FB1:  89.55  872
          ORDINAL: precision:  81.48%; recall:  90.26%; FB1:  85.64  216
              ORG: precision:  81.64%; recall:  85.46%; FB1:  83.51  1879
          PERCENT: precision:  88.42%; recall:  89.68%; FB1:  89.05  354
           PERSON: precision:  90.78%; recall:  92.10%; FB1:  91.44  2017
          PRODUCT: precision:  50.00%; recall:  68.42%; FB1:  57.78  104
         QUANTITY: precision:  66.95%; recall:  75.24%; FB1:  70.85  118
             TIME: precision:  56.95%; recall:  59.91%; FB1:  58.39  223
      WORK_OF_ART: precision:  46.53%; recall:  56.63%; FB1:  51.09  202
```