---
layout: model
title: Named Entity Recognition - CoNLL03 XLM-RoBERTa (ner_conll_xlm_roberta_base)
author: John Snow Labs
name: ner_conll_xlm_roberta_base
date: 2021-08-04
tags: [ner, en, english, conll, xlm_roberta, open_source]
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

`ner_conll_xlm_roberta_base` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_xlm_roberta_base` model is trained with`xlm_roberta_base` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_xlm_roberta_base_en_3.2.0_2.4_1628080972965.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_xlm_roberta_base_en_3.2.0_2.4_1628080972965.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = XlmRoBertaEmbeddings\
.pretrained('xlm_roberta_base', 'xx')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_xlm_roberta_base', 'en') \
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

val embeddings = XlmRoBertaEmbeddings.pretrained("xlm_roberta_base", "xx")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_xlm_roberta_base", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_xlm_roberta_base').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_xlm_roberta_base|
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

processed 46435 tokens with 5648 phrases; found: 5712 phrases; correct: 4988.
accuracy:  90.08%; (non-O)
accuracy:  97.45%; precision:  87.32%; recall:  88.31%; FB1:  87.82
LOC: precision:  88.37%; recall:  91.13%; FB1:  89.73  1720
MISC: precision:  77.43%; recall:  78.21%; FB1:  77.82  709
ORG: precision:  86.28%; recall:  84.41%; FB1:  85.33  1625
PER: precision:  91.50%; recall:  93.82%; FB1:  92.64  1658

Dev:

precision    recall  f1-score   support

B-LOC       0.95      0.95      0.95      1837
I-ORG       0.91      0.90      0.91       751
I-MISC       0.88      0.85      0.86       346
I-LOC       0.90      0.91      0.91       257
I-PER       0.97      0.98      0.98      1307
B-MISC       0.91      0.88      0.90       922
B-ORG       0.91      0.89      0.90      1341
B-PER       0.93      0.96      0.95      1842

micro avg       0.93      0.93      0.93      8603
macro avg       0.92      0.92      0.92      8603
weighted avg       0.93      0.93      0.93      8603

processed 51362 tokens with 5942 phrases; found: 5947 phrases; correct: 5483.
accuracy:  93.12%; (non-O)
accuracy:  98.55%; precision:  92.20%; recall:  92.28%; FB1:  92.24
LOC: precision:  94.48%; recall:  95.05%; FB1:  94.76  1848
MISC: precision:  89.72%; recall:  86.12%; FB1:  87.88  885
ORG: precision:  90.01%; recall:  87.99%; FB1:  88.99  1311
PER: precision:  92.64%; recall:  95.71%; FB1:  94.15  1903
```