---
layout: model
title: Named Entity Recognition - CoNLL03 BERT (ner_conll_bert_base_cased)
author: John Snow Labs
name: ner_conll_bert_base_cased
date: 2021-08-04
tags: [ner, en, english, bert, open_source, conll]
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

`ner_conll_bert_base_cased` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_bert_base_cased` model is trained with`bert_base_cased` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_bert_base_cased_en_3.2.0_2.4_1628079565109.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_bert_base_cased_en_3.2.0_2.4_1628079565109.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings\
.pretrained('bert_base_cased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_bert_base_cased', 'en') \
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

val embeddings = BertEmbeddings.pretrained("bert_base_cased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_bert_base_cased", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_bert_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_bert_base_cased|
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

B-LOC       0.94      0.92      0.93      1668
I-ORG       0.83      0.94      0.88       835
I-MISC       0.66      0.75      0.70       216
I-LOC       0.83      0.87      0.85       257
I-PER       0.98      0.98      0.98      1156
B-MISC       0.84      0.81      0.82       702
B-ORG       0.86      0.92      0.89      1661
B-PER       0.97      0.94      0.96      1617

micro avg       0.90      0.92      0.91      8112
macro avg       0.86      0.89      0.88      8112
weighted avg       0.90      0.92      0.91      8112

processed 46435 tokens with 5648 phrases; found: 5691 phrases; correct: 5101.
accuracy:  91.95%; (non-O)
accuracy:  97.90%; precision:  89.63%; recall:  90.32%; FB1:  89.97
LOC: precision:  93.10%; recall:  91.43%; FB1:  92.26  1638
MISC: precision:  80.56%; recall:  78.49%; FB1:  79.51  684
ORG: precision:  84.00%; recall:  90.73%; FB1:  87.24  1794
PER: precision:  96.38%; recall:  93.88%; FB1:  95.11  1575


Dev:

precision    recall  f1-score   support

B-LOC       0.97      0.96      0.96      1837
I-ORG       0.89      0.96      0.93       751
I-MISC       0.89      0.85      0.87       346
I-LOC       0.94      0.91      0.92       257
I-PER       0.99      0.97      0.98      1307
B-MISC       0.94      0.90      0.92       922
B-ORG       0.89      0.96      0.92      1341
B-PER       0.98      0.96      0.97      1842

micro avg       0.95      0.95      0.95      8603
macro avg       0.94      0.93      0.93      8603
weighted avg       0.95      0.95      0.95      8603

processed 51362 tokens with 5942 phrases; found: 5958 phrases; correct: 5606.
accuracy:  94.80%; (non-O)
accuracy:  98.88%; precision:  94.09%; recall:  94.35%; FB1:  94.22
LOC: precision:  96.69%; recall:  95.54%; FB1:  96.11  1815
MISC: precision:  92.41%; recall:  88.50%; FB1:  90.42  883
ORG: precision:  88.03%; recall:  94.85%; FB1:  91.31  1445
PER: precision:  97.13%; recall:  95.71%; FB1:  96.42  1815
```
