---
layout: model
title: Named Entity Recognition - CoNLL03 ALBERT Base (ner_conll_albert_base_uncased)
author: John Snow Labs
name: ner_conll_albert_base_uncased
date: 2021-08-31
tags: [ner, english, conll, albert, base, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.2.2
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`ner_conll_albert_base_uncased` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_albert_base_uncased` model is trained with`albert_base_uncased` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_albert_base_uncased_en_3.2.2_2.4_1630417931344.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_albert_base_uncased_en_3.2.2_2.4_1630417931344.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = AlbertEmbeddings\
.pretrained('albert_base_uncased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_albert_base_uncased', 'en') \
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

val embeddings = AlbertEmbeddings.pretrained("albert_base_uncased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_albert_base_uncased", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_albert_base_uncased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_albert_base_uncased|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.2+|
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

B-LOC       0.92      0.90      0.91      1668
I-ORG       0.81      0.87      0.84       835
I-MISC       0.61      0.62      0.62       216
I-LOC       0.82      0.77      0.79       257
I-PER       0.97      0.99      0.98      1156
B-MISC       0.80      0.77      0.79       702
B-ORG       0.86      0.86      0.86      1661
B-PER       0.94      0.96      0.95      1617

micro avg       0.89      0.89      0.89      8112
macro avg       0.84      0.84      0.84      8112
weighted avg       0.89      0.89      0.89      8112


processed 46435 tokens with 5648 phrases; found: 5631 phrases; correct: 4955.
accuracy:  89.23%; (non-O)
accuracy:  97.33%; precision:  88.00%; recall:  87.73%; FB1:  87.86
LOC: precision:  91.30%; recall:  89.39%; FB1:  90.34  1633
MISC: precision:  76.66%; recall:  73.93%; FB1:  75.27  677
ORG: precision:  84.09%; recall:  83.99%; FB1:  84.04  1659
PER: precision:  93.26%; recall:  95.86%; FB1:  94.54  1662

Dev:


precision    recall  f1-score   support

B-LOC       0.96      0.96      0.96      1837
I-ORG       0.89      0.86      0.87       751
I-MISC       0.89      0.71      0.79       346
I-LOC       0.92      0.88      0.90       257
I-PER       0.97      0.98      0.98      1307
B-MISC       0.90      0.87      0.88       922
B-ORG       0.92      0.90      0.91      1341
B-PER       0.96      0.98      0.97      1842

micro avg       0.94      0.93      0.93      8603
macro avg       0.93      0.89      0.91      8603
weighted avg       0.94      0.93      0.93      8603


processed 51362 tokens with 5942 phrases; found: 5927 phrases; correct: 5493.
accuracy:  92.62%; (non-O)
accuracy:  98.40%; precision:  92.68%; recall:  92.44%; FB1:  92.56
LOC: precision:  95.00%; recall:  95.21%; FB1:  95.11  1841
MISC: precision:  87.77%; recall:  84.06%; FB1:  85.87  883
ORG: precision:  88.96%; recall:  87.10%; FB1:  88.02  1313
PER: precision:  95.29%; recall:  97.77%; FB1:  96.52  1890
```