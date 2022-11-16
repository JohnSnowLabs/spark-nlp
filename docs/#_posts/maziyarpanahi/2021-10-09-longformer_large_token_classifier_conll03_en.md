---
layout: model
title: Longformer Token Classification Base - NER CoNLL (longformer_large_token_classifier_conll03)
author: John Snow Labs
name: longformer_large_token_classifier_conll03
date: 2021-10-09
tags: [token_classification, open_source, ner, en, english, longformer, conll]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: LongformerForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`Longformer Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

**longformer_large_token_classifier_conll03** is a fine-tuned Longformer model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFLongformerForTokenClassification](https://huggingface.co/transformers/model_doc/longformer.html#tflongformerfortokenclassification) to train this model and used `LongformerForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities
`PER`, `LOC`, `ORG`, `MISC`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_large_token_classifier_conll03_en_3.3.0_3.0_1633778673017.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = LongformerForTokenClassification \
      .pretrained('longformer_large_token_classifier_conll03', 'en') \
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

val tokenClassifier = LongformerForTokenClassification.pretrained("longformer_large_token_classifier_conll03", "en")
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
|Model Name:|longformer_large_token_classifier_conll03|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|4096|

## Data Source

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
            precision    recall  f1-score   support

       B-LOC       0.97      0.92      0.94      1837
      B-MISC       0.93      0.88      0.90       922
       B-ORG       0.91      0.92      0.92      1341
       B-PER       0.96      0.98      0.97      1842
       I-LOC       0.91      0.92      0.92       257
      I-MISC       0.89      0.82      0.85       346
       I-ORG       0.90      0.94      0.92       751
       I-PER       0.98      0.98      0.98      1307
           O       1.00      1.00      1.00     42759

    accuracy                           0.99     51362
   macro avg       0.94      0.93      0.93     51362
weighted avg       0.99      0.99      0.99     51362



processed 51362 tokens with 5942 phrases; found: 5900 phrases; correct: 5504.
accuracy:  93.75%; (non-O)
accuracy:  98.70%; precision:  93.29%; recall:  92.63%; FB1:  92.96
              LOC: precision:  96.19%; recall:  92.00%; FB1:  94.05  1757
             MISC: precision:  90.28%; recall:  86.66%; FB1:  88.43  885
              ORG: precision:  88.81%; recall:  90.53%; FB1:  89.66  1367
              PER: precision:  95.24%; recall:  97.77%; FB1:  96.49  1891
              
```
