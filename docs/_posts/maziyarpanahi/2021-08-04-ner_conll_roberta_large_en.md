---
layout: model
title: Named Entity Recognition - CoNLL03 RoBERTa (ner_conll_roberta_large)
author: John Snow Labs
name: ner_conll_roberta_large
date: 2021-08-04
tags: [ner, en, english, roberta, open_source, conll]
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

`ner_conll_roberta_large` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_roberta_large` model is trained with`roberta_large` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_roberta_large_en_3.2.0_2.4_1628080747001.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_roberta_large_en_3.2.0_2.4_1628080747001.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
.pretrained('roberta_large', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_roberta_large', 'en') \
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

val embeddings = RoBertaEmbeddings.pretrained("roberta_large", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_roberta_large", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_roberta_large').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_roberta_large|
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

B-LOC       0.93      0.92      0.92      1668
I-ORG       0.87      0.91      0.89       835
I-MISC       0.57      0.77      0.66       216
I-LOC       0.89      0.86      0.88       257
I-PER       0.97      0.98      0.98      1156
B-MISC       0.76      0.85      0.80       702
B-ORG       0.89      0.90      0.90      1661
B-PER       0.96      0.95      0.95      1617

micro avg       0.90      0.92      0.91      8112
macro avg       0.86      0.89      0.87      8112
weighted avg       0.90      0.92      0.91      8112

processed 46435 tokens with 5648 phrases; found: 5719 phrases; correct: 5109.
accuracy:  91.80%; (non-O)
accuracy:  97.87%; precision:  89.33%; recall:  90.46%; FB1:  89.89
LOC: precision:  92.41%; recall:  91.19%; FB1:  91.79  1646
MISC: precision:  73.83%; recall:  83.19%; FB1:  78.23  791
ORG: precision:  87.43%; recall:  88.74%; FB1:  88.08  1686
PER: precision:  95.86%; recall:  94.62%; FB1:  95.24  1596


Dev:

precision    recall  f1-score   support

B-LOC       0.97      0.96      0.96      1837
I-ORG       0.95      0.92      0.93       751
I-MISC       0.87      0.89      0.88       346
I-LOC       0.94      0.93      0.94       257
I-PER       0.98      0.98      0.98      1307
B-MISC       0.87      0.93      0.90       922
B-ORG       0.94      0.94      0.94      1341
B-PER       0.97      0.98      0.97      1842

micro avg       0.95      0.95      0.95      8603
macro avg       0.94      0.94      0.94      8603
weighted avg       0.95      0.95      0.95      8603

processed 51362 tokens with 5942 phrases; found: 5996 phrases; correct: 5633.
accuracy:  95.18%; (non-O)
accuracy:  98.95%; precision:  93.95%; recall:  94.80%; FB1:  94.37
LOC: precision:  96.75%; recall:  95.65%; FB1:  96.19  1816
MISC: precision:  85.61%; recall:  91.65%; FB1:  88.53  987
ORG: precision:  92.14%; recall:  92.62%; FB1:  92.38  1348
PER: precision:  96.96%; recall:  97.12%; FB1:  97.04  1845


```