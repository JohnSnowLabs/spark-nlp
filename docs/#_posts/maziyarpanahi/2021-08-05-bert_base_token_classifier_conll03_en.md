---
layout: model
title: BERT Token Classification - NER CoNLL (bert_base_token_classifier_conll03)
author: John Snow Labs
name: bert_base_token_classifier_conll03
date: 2021-08-05
tags: [ner, conll, en, english, token_classification, bert, open_source]
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


**bert_base_token_classifier_conll03** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFBertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html#tfbertfortokenclassification) to train this model and used `BertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_token_classifier_conll03_en_3.2.0_2.4_1628165842529.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_base_token_classifier_conll03', 'en') \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_base_token_classifier_conll03", "en")
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
nlu.load("en.classify.token_bert.conll03").predict("""My name is John!""")
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
|Model Name:|bert_base_token_classifier_conll03|
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
Test:

precision    recall  f1-score   support

B-LOC       0.94      0.90      0.92      1668
I-ORG       0.85      0.93      0.88       835
I-MISC       0.63      0.80      0.71       216
I-LOC       0.87      0.84      0.86       257
I-PER       0.98      0.98      0.98      1156
B-MISC       0.78      0.82      0.80       702
B-ORG       0.88      0.91      0.89      1661
B-PER       0.96      0.94      0.95      1617

micro avg       0.90      0.91      0.91      8112
macro avg       0.86      0.89      0.87      8112
weighted avg       0.90      0.91      0.91      8112



processed 46435 tokens with 5648 phrases; found: 5730 phrases; correct: 5050.
accuracy:  91.33%; (non-O)
accuracy:  97.83%; precision:  88.13%; recall:  89.41%; FB1:  88.77
LOC: precision:  92.57%; recall:  89.69%; FB1:  91.11  1616
MISC: precision:  71.92%; recall:  79.91%; FB1:  75.71  780
ORG: precision:  84.89%; recall:  88.92%; FB1:  86.86  1740
PER: precision:  95.11%; recall:  93.75%; FB1:  94.43  1594


Dev:

precision    recall  f1-score   support

B-LOC       0.96      0.91      0.93      1837
I-ORG       0.90      0.94      0.92       751
I-MISC       0.83      0.84      0.84       346
I-LOC       0.92      0.93      0.93       257
I-PER       0.99      0.98      0.98      1307
B-MISC       0.88      0.90      0.89       922
B-ORG       0.90      0.92      0.91      1341
B-PER       0.97      0.97      0.97      1842

micro avg       0.94      0.93      0.93      8603
macro avg       0.92      0.92      0.92      8603
weighted avg       0.94      0.93      0.93      8603



processed 51362 tokens with 5942 phrases; found: 5961 phrases; correct: 5457.
accuracy:  93.33%; (non-O)
accuracy:  98.64%; precision:  91.55%; recall:  91.84%; FB1:  91.69
LOC: precision:  95.09%; recall:  90.64%; FB1:  92.81  1751
MISC: precision:  83.45%; recall:  87.53%; FB1:  85.44  967
ORG: precision:  86.43%; recall:  90.75%; FB1:  88.54  1408
PER: precision:  96.35%; recall:  95.98%; FB1:  96.17  1835

```