---
layout: model
title: Named Entity Recognition - CoNLL03 DistilBERT (ner_conll_distilbert_base_cased)
author: John Snow Labs
name: ner_conll_distilbert_base_cased
date: 2021-08-04
tags: [en, english, ner, distilbert, open_source, conll]
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

`ner_conll_distilbert_base_cased` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_distilbert_base_cased` model is trained with`distilbert_base_cased` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_distilbert_base_cased_en_3.2.0_2.4_1628079967124.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_distilbert_base_cased_en_3.2.0_2.4_1628079967124.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DistilBertEmbeddings\
.pretrained('distilbert_base_cased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_distilbert_base_cased', 'en') \
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

val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_distilbert_base_cased", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_distilbert_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_distilbert_base_cased|
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

B-LOC       0.92      0.90      0.91      1668
I-ORG       0.85      0.88      0.86       835
I-MISC       0.61      0.73      0.67       216
I-LOC       0.82      0.81      0.81       257
I-PER       0.97      0.99      0.98      1156
B-MISC       0.78      0.79      0.79       702
B-ORG       0.86      0.87      0.87      1661
B-PER       0.94      0.95      0.94      1617

micro avg       0.89      0.90      0.89      8112
macro avg       0.84      0.87      0.85      8112
weighted avg       0.89      0.90      0.89      8112

processed 46435 tokens with 5648 phrases; found: 5663 phrases; correct: 4971.
accuracy:  89.83%; (non-O)
accuracy:  97.54%; precision:  87.78%; recall:  88.01%; FB1:  87.90
LOC: precision:  91.77%; recall:  89.57%; FB1:  90.66  1628
MISC: precision:  74.93%; recall:  76.64%; FB1:  75.77  718
ORG: precision:  83.99%; recall:  85.25%; FB1:  84.61  1686
PER: precision:  93.38%; recall:  94.19%; FB1:  93.78  1631


Dev:

precision    recall  f1-score   support

B-LOC       0.96      0.94      0.95      1837
I-ORG       0.92      0.86      0.89       751
I-MISC       0.83      0.88      0.85       346
I-LOC       0.91      0.90      0.90       257
I-PER       0.97      0.97      0.97      1307
B-MISC       0.89      0.90      0.90       922
B-ORG       0.91      0.90      0.91      1341
B-PER       0.96      0.96      0.96      1842

micro avg       0.94      0.93      0.93      8603
macro avg       0.92      0.91      0.92      8603
weighted avg       0.94      0.93      0.93      8603


processed 51362 tokens with 5942 phrases; found: 5918 phrases; correct: 5492.
accuracy:  92.92%; (non-O)
accuracy:  98.58%; precision:  92.80%; recall:  92.43%; FB1:  92.61
LOC: precision:  95.68%; recall:  94.01%; FB1:  94.84  1805
MISC: precision:  87.03%; recall:  88.07%; FB1:  87.55  933
ORG: precision:  89.46%; recall:  88.59%; FB1:  89.02  1328
PER: precision:  95.30%; recall:  95.82%; FB1:  95.56  1852
```