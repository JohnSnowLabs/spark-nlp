---
layout: model
title: XLM-RoBERTa Token Classification Base - NER OntoNotes (xlm_roberta_base_token_classifier_ontonotes)
author: John Snow Labs
name: xlm_roberta_base_token_classifier_ontonotes
date: 2021-10-03
tags: [ontonotes, token_classification, ner, en, english, xlm_roberta, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`XLM-RoBERTa Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.


**xlm_roberta_base_token_classifier_ontonotes** is a fine-tuned XLM-RoBERTa model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, and WORK_OF_ART.

We used [TFXLMRobertaForTokenClassification](https://huggingface.co/transformers/model_doc/xlmroberta.html#tfxlmrobertafortokenclassification) to train this model and used `RoBertaForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_token_classifier_ontonotes_en_3.3.0_3.0_1633271114226.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_token_classifier_ontonotes_en_3.3.0_3.0_1633271114226.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = XlmRoBertaForTokenClassification \
      .pretrained('xlm_roberta_base_token_classifier_ontonotes', 'en') \
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

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_base_token_classifier_ontonotes", "en")
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
|Model Name:|xlm_roberta_base_token_classifier_ontonotes|
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

   B-CARDINAL       0.86      0.86      0.86       935
       B-DATE       0.88      0.90      0.89      1602
      B-EVENT       0.64      0.51      0.57        63
        B-FAC       0.73      0.75      0.74       135
        B-GPE       0.97      0.96      0.96      2240
   B-LANGUAGE       0.80      0.55      0.65        22
        B-LAW       0.78      0.62      0.69        40
        B-LOC       0.78      0.80      0.79       179
      B-MONEY       0.87      0.91      0.89       314
       B-NORP       0.94      0.96      0.95       841
    B-ORDINAL       0.81      0.92      0.86       195
        B-ORG       0.92      0.90      0.91      1795
    B-PERCENT       0.93      0.95      0.94       349
     B-PERSON       0.94      0.96      0.95      1988
    B-PRODUCT       0.74      0.75      0.75        76
   B-QUANTITY       0.79      0.82      0.80       105
       B-TIME       0.72      0.66      0.68       212
B-WORK_OF_ART       0.68      0.72      0.70       166
   I-CARDINAL       0.79      0.89      0.83       331
       I-DATE       0.90      0.92      0.91      2011
      I-EVENT       0.70      0.81      0.75       130
        I-FAC       0.84      0.86      0.85       213
        I-GPE       0.94      0.92      0.93       628
        I-LAW       0.84      0.65      0.73       106
        I-LOC       0.85      0.83      0.84       180
      I-MONEY       0.94      0.94      0.94       685
       I-NORP       0.95      0.92      0.94       160
    I-ORDINAL       0.00      0.00      0.00         4
        I-ORG       0.92      0.93      0.93      2406
    I-PERCENT       0.95      0.96      0.96       523
     I-PERSON       0.95      0.96      0.96      1412
    I-PRODUCT       0.73      0.78      0.76        69
   I-QUANTITY       0.83      0.93      0.88       206
       I-TIME       0.70      0.77      0.74       255
I-WORK_OF_ART       0.70      0.68      0.69       337
            O       0.99      0.99      0.99    131815

     accuracy                           0.98    152728
    macro avg       0.81      0.81      0.81    152728
 weighted avg       0.98      0.98      0.98    152728



processed 152728 tokens with 11257 phrases; found: 11498 phrases; correct: 10038.
accuracy:  90.96%; (non-O)
accuracy:  98.12%; precision:  87.30%; recall:  89.17%; FB1:  88.23
         CARDINAL: precision:  82.65%; recall:  84.06%; FB1:  83.35  951
             DATE: precision:  83.75%; recall:  87.52%; FB1:  85.59  1674
            EVENT: precision:  56.90%; recall:  52.38%; FB1:  54.55  58
              FAC: precision:  69.44%; recall:  74.07%; FB1:  71.68  144
              GPE: precision:  95.46%; recall:  94.87%; FB1:  95.16  2226
         LANGUAGE: precision:  80.00%; recall:  54.55%; FB1:  64.86  15
              LAW: precision:  66.67%; recall:  60.00%; FB1:  63.16  36
              LOC: precision:  70.35%; recall:  78.21%; FB1:  74.07  199
            MONEY: precision:  81.63%; recall:  86.31%; FB1:  83.90  332
             NORP: precision:  92.51%; recall:  95.48%; FB1:  93.97  868
          ORDINAL: precision:  81.36%; recall:  91.79%; FB1:  86.27  220
              ORG: precision:  87.76%; recall:  87.08%; FB1:  87.42  1781
          PERCENT: precision:  87.68%; recall:  89.68%; FB1:  88.67  357
           PERSON: precision:  93.19%; recall:  95.62%; FB1:  94.39  2040
          PRODUCT: precision:  67.90%; recall:  72.37%; FB1:  70.06  81
         QUANTITY: precision:  69.83%; recall:  77.14%; FB1:  73.30  116
             TIME: precision:  62.79%; recall:  63.68%; FB1:  63.23  215
      WORK_OF_ART: precision:  62.16%; recall:  69.28%; FB1:  65.53  185
```