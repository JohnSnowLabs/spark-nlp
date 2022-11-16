---
layout: model
title: RoBERTa Token Classification Base - NER CoNLL (roberta_base_token_classifier_conll03)
author: John Snow Labs
name: roberta_base_token_classifier_conll03
date: 2021-09-26
tags: [roberta, ner, en, english, open_source, token_classification, conll]
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

**roberta_base_token_classifier_conll03** is a fine-tuned RoBERTa model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFRobertaForTokenClassification](https://huggingface.co/transformers/model_doc/roberta.html#tfrobertafortokenclassification) to train this model and used `RoBertaForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_token_classifier_conll03_en_3.3.0_3.0_1632677482956.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('roberta_base_token_classifier_conll03', 'en') \
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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_base_token_classifier_conll03", "en")
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
nlu.load("en.classify.token_roberta_base_token_classifier_conll03").predict("""My name is John!""")
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
|Model Name:|roberta_base_token_classifier_conll03|
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

B-LOC       0.94      0.86      0.90      1837
I-ORG       0.86      0.91      0.88       751
I-MISC       0.85      0.76      0.80       346
I-LOC       0.85      0.77      0.81       257
I-PER       0.98      0.97      0.98      1307
B-MISC       0.88      0.84      0.86       922
B-ORG       0.85      0.91      0.88      1341
B-PER       0.95      0.95      0.95      1842

micro avg       0.91      0.90      0.91      8603
macro avg       0.90      0.87      0.88      8603
weighted avg       0.92      0.90      0.91      8603



processed 51362 tokens with 5942 phrases; found: 5928 phrases; correct: 5270.
accuracy:  90.04%; (non-O)
accuracy:  98.07%; precision:  88.90%; recall:  88.69%; FB1:  88.80
LOC: precision:  92.54%; recall:  85.08%; FB1:  88.66  1689
MISC: precision:  83.48%; recall:  81.67%; FB1:  82.57  902
ORG: precision:  81.28%; recall:  90.01%; FB1:  85.42  1485
PER: precision:  94.33%; recall:  94.84%; FB1:  94.59  1852
```