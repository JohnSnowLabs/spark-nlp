---
layout: model
title: Named Entity Recognition - CoNLL03 RoBERTa (ner_conll_roberta_base)
author: John Snow Labs
name: ner_conll_roberta_base
date: 2021-08-04
tags: [ner, en, english, conll, roberta, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`ner_conll_roberta_base` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_roberta_base` model is trained with`roberta_base` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_roberta_base_en_3.2.0_2.4_1628080425702.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_roberta_base_en_3.2.0_2.4_1628080425702.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings\
.pretrained('roberta_base', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_roberta_base', 'en') \
.setInputCols(['document', 'token', 'embeddings']) \
.setOutputCol('ner')

ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
embeddings,
ner_model,
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

val embeddings = RoBertaEmbeddings.pretrained("roberta_base", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_roberta_base", "en") 
.setInputCols("document"', "token", "embeddings") 
.setOutputCol("ner")

val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner_model, ner_converter))

val example = Seq.empty["My name is John!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```

{:.nlu-block}
```python
import nlu

text = ["My name is John!"]

ner_df = nlu.load('en.ner.ner_conll_roberta_base').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_roberta_base|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
Test:

precision    recall  f1-score   support

B-LOC       0.90      0.92      0.91      1668
I-ORG       0.85      0.90      0.87       835
I-MISC       0.61      0.73      0.67       216
I-LOC       0.80      0.87      0.84       257
I-PER       0.98      0.98      0.98      1156
B-MISC       0.82      0.81      0.82       702
B-ORG       0.87      0.88      0.87      1661
B-PER       0.95      0.94      0.94      1617

micro avg       0.89      0.90      0.90      8112
macro avg       0.85      0.88      0.86      8112
weighted avg       0.89      0.90      0.90      8112

processed 46435 tokens with 5648 phrases; found: 5675 phrases; correct: 5027.
accuracy:  90.50%; (non-O)
accuracy:  97.69%; precision:  88.58%; recall:  89.00%; FB1:  88.79
LOC: precision:  89.89%; recall:  91.13%; FB1:  90.50  1691
MISC: precision:  78.86%; recall:  78.63%; FB1:  78.74  700
ORG: precision:  85.65%; recall:  86.27%; FB1:  85.96  1673
PER: precision:  94.48%; recall:  94.12%; FB1:  94.30  1611

Dev:

precision    recall  f1-score   support

B-LOC       0.94      0.96      0.95      1837
I-ORG       0.93      0.91      0.92       751
I-MISC       0.86      0.85      0.85       346
I-LOC       0.94      0.92      0.93       257
I-PER       0.98      0.97      0.98      1307
B-MISC       0.92      0.88      0.90       922
B-ORG       0.91      0.93      0.92      1341
B-PER       0.96      0.97      0.97      1842

micro avg       0.94      0.94      0.94      8603
macro avg       0.93      0.92      0.93      8603
weighted avg       0.94      0.94      0.94      8603

processed 51362 tokens with 5942 phrases; found: 5959 phrases; correct: 5544.
accuracy:  93.94%; (non-O)
accuracy:  98.76%; precision:  93.04%; recall:  93.30%; FB1:  93.17
LOC: precision:  94.35%; recall:  95.48%; FB1:  94.91  1859
MISC: precision:  90.19%; recall:  86.77%; FB1:  88.45  887
ORG: precision:  89.24%; recall:  90.90%; FB1:  90.06  1366
PER: precision:  95.89%; recall:  96.15%; FB1:  96.02  1847
```