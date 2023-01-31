---
layout: model
title: XLNet Token Classification Base - NER CoNLL (xlnet_base_token_classifier_conll03)
author: John Snow Labs
name: xlnet_base_token_classifier_conll03
date: 2021-09-28
tags: [ner, en, english, open_source, xlnet, base, token_classification]
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

**xlnet_base_token_classifier_conll03** is a fine-tuned XLNet model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). 

We used [TFXLNetForTokenClassification](https://huggingface.co/transformers/model_doc/xlnet.html#tfxlnetfortokenclassification) to train this model and used `XlnetForTokenClassification` annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_base_token_classifier_conll03_en_3.3.0_3.0_1632831424304.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlnet_base_token_classifier_conll03_en_3.3.0_3.0_1632831424304.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
.pretrained('xlnet_base_token_classifier_conll03', 'en') \
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

val tokenClassifier = XlnetForTokenClassification.pretrained("xlnet_base_token_classifier_conll03", "en")
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
nlu.load("en.classify.token_xlnet_base_token_classifier_conll03").predict("""My name is John!""")
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
|Model Name:|xlnet_base_token_classifier_conll03|
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

B-LOC       0.94      0.97      0.96      1837
B-MISC       0.84      0.92      0.88       922
B-ORG       0.89      0.94      0.91      1341
B-PER       0.96      0.97      0.97      1842
I-LOC       0.92      0.93      0.92       257
I-MISC       0.84      0.81      0.82       346
I-ORG       0.93      0.89      0.91       751
I-PER       0.96      0.98      0.97      1307
O       1.00      0.99      1.00     42759

accuracy                           0.98     51362
macro avg       0.92      0.93      0.93     51362
weighted avg       0.99      0.98      0.99     51362



processed 51362 tokens with 5942 phrases; found: 6219 phrases; correct: 5550.
accuracy:  94.57%; (non-O)
accuracy:  98.49%; precision:  89.24%; recall:  93.40%; FB1:  91.28
LOC: precision:  93.30%; recall:  96.30%; FB1:  94.78  1896
MISC: precision:  79.47%; recall:  88.18%; FB1:  83.60  1023
ORG: precision:  84.71%; recall:  90.08%; FB1:  87.31  1426
PER: precision:  93.92%; recall:  95.55%; FB1:  94.73  1874
```