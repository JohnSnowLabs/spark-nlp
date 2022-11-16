---
layout: model
title: DistilBERT Token Classification - NER CoNLL (distilbert_base_token_classifier_conll03)
author: John Snow Labs
name: distilbert_base_token_classifier_conll03
date: 2021-08-05
tags: [open_source, ner, en, english, token_classification, distilbert]
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


**distilbert_base_token_classifier_conll03** is a fine-tuned DistilBERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFDistilBertForTokenClassification](https://huggingface.co/transformers/model_doc/distilbert.html#tfdistilbertfortokenclassification) to train this model and used `DistilBertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_token_classifier_conll03_en_3.2.0_2.4_1628173085206.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('distilbert_base_token_classifier_conll03', 'en') \
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

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_base_token_classifier_conll03", "en")
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
 |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
 +------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_token_classifier_conll03|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
Dev:

         precision    recall  f1-score   support

       B-LOC       0.95      0.88      0.91      1837
      B-MISC       0.85      0.86      0.85       922
       B-ORG       0.86      0.90      0.88      1341
       B-PER       0.96      0.96      0.96      1842
       I-LOC       0.87      0.91      0.89       257
      I-MISC       0.80      0.80      0.80       346
       I-ORG       0.87      0.92      0.90       751
       I-PER       0.98      0.97      0.98      1307
           O       1.00      1.00      1.00     42759

    accuracy                           0.98     51362
   macro avg       0.90      0.91      0.91     51362
weighted avg       0.98      0.98      0.98     51362



processed 51362 tokens with 5942 phrases; found: 5973 phrases; correct: 5306.
accuracy:  91.36%; (non-O)
accuracy:  98.20%; precision:  88.83%; recall:  89.30%; FB1:  89.06
              LOC: precision:  94.11%; recall:  87.81%; FB1:  90.85  1714
             MISC: precision:  78.38%; recall:  81.78%; FB1:  80.04  962
              ORG: precision:  82.01%; recall:  88.07%; FB1:  84.93  1440
              PER: precision:  94.67%; recall:  95.44%; FB1:  95.05  1857


Test:


        precision    recall  f1-score   support

       B-LOC       0.93      0.85      0.89      1668
      B-MISC       0.77      0.78      0.78       702
       B-ORG       0.81      0.89      0.85      1661
       B-PER       0.95      0.93      0.94      1617
       I-LOC       0.80      0.76      0.78       257
      I-MISC       0.60      0.69      0.64       216
       I-ORG       0.80      0.92      0.86       835
       I-PER       0.98      0.98      0.98      1156
           O       0.99      0.99      0.99     38323

    accuracy                           0.97     46435
   macro avg       0.85      0.87      0.86     46435
weighted avg       0.97      0.97      0.97     46435



processed 46435 tokens with 5648 phrases; found: 5738 phrases; correct: 4864.
accuracy:  88.52%; (non-O)
accuracy:  97.24%; precision:  84.77%; recall:  86.12%; FB1:  85.44
              LOC: precision:  91.36%; recall:  84.29%; FB1:  87.68  1539
             MISC: precision:  70.60%; recall:  75.93%; FB1:  73.16  755
              ORG: precision:  77.29%; recall:  86.27%; FB1:  81.54  1854
              PER: precision:  93.84%; recall:  92.27%; FB1:  93.05  1590
```
