---
layout: model
title: XLNet Token Classification Large - NER CoNLL (xlnet_large_token_classifier_conll03)
author: John Snow Labs
name: xlnet_large_token_classifier_conll03
date: 2021-09-28
tags: [ner, xlnet, large, open_source, en, english, token_classification]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: XlnetForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`XLNet Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

**xlnet_large_token_classifier_conll03** is a fine-tuned XLNet model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFXLNetForTokenClassification](https://huggingface.co/transformers/model_doc/xlnet.html#tfxlnetfortokenclassification) to train this model and used `XlnetForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_large_token_classifier_conll03_en_3.3.0_3.0_1632832610836.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlnet_large_token_classifier_conll03_en_3.3.0_3.0_1632832610836.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = XlnetForTokenClassification \
.pretrained('xlnet_large_token_classifier_conll03', 'en') \
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

val tokenClassifier = XlnetForTokenClassification.pretrained("xlnet_large_token_classifier_conll03", "en")
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
nlu.load("en.classify.token_xlnet.large_token_classifier_conll03").predict("""My name is John!""")
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
|Model Name:|xlnet_large_token_classifier_conll03|
|Compatibility:|Spark NLP 3.3.0+|
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
precision    recall  f1-score   support

B-LOC       0.95      0.96      0.96      1837
B-MISC       0.83      0.93      0.87       922
B-ORG       0.90      0.93      0.91      1341
B-PER       0.97      0.97      0.97      1842
I-LOC       0.93      0.93      0.93       257
I-MISC       0.86      0.83      0.84       346
I-ORG       0.90      0.89      0.89       751
I-PER       0.96      0.98      0.97      1307
O       1.00      0.99      1.00     42759

accuracy                           0.99     51362
macro avg       0.92      0.93      0.93     51362
weighted avg       0.99      0.99      0.99     51362



processed 51362 tokens with 5942 phrases; found: 6168 phrases; correct: 5552.
accuracy:  94.56%; (non-O)
accuracy:  98.57%; precision:  90.01%; recall:  93.44%; FB1:  91.69
LOC: precision:  94.46%; recall:  95.65%; FB1:  95.05  1860
MISC: precision:  79.41%; recall:  89.91%; FB1:  84.33  1044
ORG: precision:  85.47%; recall:  90.38%; FB1:  87.86  1418
PER: precision:  95.02%; recall:  95.22%; FB1:  95.12  1846
```