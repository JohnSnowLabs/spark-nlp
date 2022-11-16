---
layout: model
title: BERT Token Classification Large - NER CoNLL (bert_large_token_classifier_conll03)
author: John Snow Labs
name: bert_large_token_classifier_conll03
date: 2021-08-05
tags: [ner, conll, en, english, token_classification, bert, open_source, large]
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


**bert_large_token_classifier_conll03** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFBertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html#tfbertfortokenclassification) to train this model and used `BertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_token_classifier_conll03_en_3.2.0_2.4_1628171471927.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_large_token_classifier_conll03', 'en') \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_large_token_classifier_conll03", "en")
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
nlu.load("en.classify.token_bert.large_conll03").predict("""My name is John!""")
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
|Model Name:|bert_large_token_classifier_conll03|
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

B-LOC       0.96      0.90      0.93      1837
I-ORG       0.93      0.95      0.94       751
I-MISC       0.91      0.87      0.89       346
I-LOC       0.91      0.94      0.93       257
I-PER       0.99      0.98      0.99      1307
B-MISC       0.94      0.90      0.92       922
B-ORG       0.88      0.95      0.91      1341
B-PER       0.98      0.98      0.98      1842

micro avg       0.95      0.94      0.94      8603
macro avg       0.94      0.93      0.93      8603
weighted avg       0.95      0.94      0.94      8603



processed 51362 tokens with 5942 phrases; found: 5915 phrases; correct: 5497.
accuracy:  93.99%; (non-O)
accuracy:  98.80%; precision:  92.93%; recall:  92.51%; FB1:  92.72
LOC: precision:  95.25%; recall:  89.49%; FB1:  92.28  1726
MISC: precision:  90.35%; recall:  88.39%; FB1:  89.36  902
ORG: precision:  86.75%; recall:  93.21%; FB1:  89.86  1441
PER: precision:  96.86%; recall:  97.07%; FB1:  96.96  1846


Test:

precision    recall  f1-score   support

B-LOC       0.93      0.89      0.91      1668
I-ORG       0.86      0.94      0.90       835
I-MISC       0.68      0.75      0.71       216
I-LOC       0.87      0.86      0.87       257
I-PER       0.98      0.98      0.98      1156
B-MISC       0.84      0.82      0.83       702
B-ORG       0.87      0.92      0.90      1661
B-PER       0.97      0.96      0.97      1617

micro avg       0.91      0.92      0.91      8112
macro avg       0.88      0.89      0.88      8112
weighted avg       0.91      0.92      0.92      8112



processed 46435 tokens with 5648 phrases; found: 5682 phrases; correct: 5104.
accuracy:  92.01%; (non-O)
accuracy:  98.09%; precision:  89.83%; recall:  90.37%; FB1:  90.10
LOC: precision:  92.39%; recall:  88.85%; FB1:  90.59  1604
MISC: precision:  79.75%; recall:  80.20%; FB1:  79.97  706
ORG: precision:  85.50%; recall:  91.27%; FB1:  88.29  1773
PER: precision:  96.50%; recall:  95.42%; FB1:  95.96  1599

```