---
layout: model
title: ALBERT Token Classification XLarge - NER CoNLL (albert_xlarge_token_classifier_conll03)
author: John Snow Labs
name: albert_xlarge_token_classifier_conll03
date: 2021-09-26
tags: [ner, albert, xlarge, conll, en, english, token_classification, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: AlBertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`ALBERT Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

**albert_xlarge_token_classifier_conll03** is a fine-tuned ALBERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFAlbertForTokenClassification](https://huggingface.co/transformers/model_doc/albert.html#tfalbertfortokenclassification) to train this model and used `AlbertForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities
`LOC`, `ORG`, `PER`, `MISC`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_xlarge_token_classifier_conll03_en_3.3.0_3.0_1632673242835.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_xlarge_token_classifier_conll03_en_3.3.0_3.0_1632673242835.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = AlbertForTokenClassification \
.pretrained('albert_xlarge_token_classifier_conll03', 'en') \
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

val tokenClassifier = AlbertForTokenClassification.pretrained("albert_xlarge_token_classifier_conll03", "en")
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
nlu.load("en.classify.token_albert.xlarge_token_classifier_conll03").predict("""My name is John!""")
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
|Model Name:|albert_xlarge_token_classifier_conll03|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
precision    recall  f1-score   support

B-LOC       0.96      0.97      0.97      1837
B-MISC       0.89      0.90      0.90       922
B-ORG       0.88      0.94      0.91      1341
B-PER       0.92      0.97      0.94      1842
I-LOC       0.94      0.88      0.91       257
I-MISC       0.88      0.77      0.82       346
I-ORG       0.89      0.87      0.88       751
I-PER       0.98      0.91      0.94      1307
O       1.00      0.99      0.99     42759

accuracy                           0.98     51362
macro avg       0.93      0.91      0.92     51362
weighted avg       0.98      0.98      0.98     51362



processed 51362 tokens with 5942 phrases; found: 6183 phrases; correct: 5466.
accuracy:  92.84%; (non-O)
accuracy:  98.33%; precision:  88.40%; recall:  91.99%; FB1:  90.16
LOC: precision:  95.47%; recall:  96.35%; FB1:  95.91  1854
MISC: precision:  85.05%; recall:  87.64%; FB1:  86.32  950
ORG: precision:  84.29%; recall:  90.45%; FB1:  87.27  1439
PER: precision:  86.34%; recall:  90.93%; FB1:  88.58  1940
```