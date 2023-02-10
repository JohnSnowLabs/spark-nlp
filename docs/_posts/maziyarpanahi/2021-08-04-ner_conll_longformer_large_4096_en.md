---
layout: model
title: Named Entity Recognition - CoNLL03 Longformer (ner_conll_longformer_large_4096)
author: John Snow Labs
name: ner_conll_longformer_large_4096
date: 2021-08-04
tags: [en, english, ner, longformer, open_source, conll]
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

`ner_conll_longformer_large_4096` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_longformer_large_4096` model has trained with`longformer_large_4096` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_longformer_large_4096_en_3.2.0_2.4_1628082029556.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_longformer_large_4096_en_3.2.0_2.4_1628082029556.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = LongformerEmbeddings\
.pretrained("longformer_large_4096")\
.setInputCols(['document', 'token'])\
.setOutputCol("embeddings")\
.setCaseSensitive(True)\
.setMaxSentenceLength(4096)

ner_model = NerDLModel.pretrained('ner_conll_longformer_large_4096', 'en') \
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

val embeddings = LongformerEmbeddings.pretrained("longformer_large_4096", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")
.setCaseSensitive(true)
.setMaxSentenceLength(4096)

val ner_model = NerDLModel.pretrained("ner_conll_longformer_large_4096", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_longformer_large_4096').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_longformer_large_4096|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

[https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Benchmarking

```bash
Test:

precision    recall  f1-score   support

B-LOC       0.95      0.93      0.94      1668
I-ORG       0.88      0.92      0.90       835
I-MISC       0.53      0.81      0.64       216
I-LOC       0.85      0.90      0.87       257
I-PER       0.97      0.99      0.98      1156
B-MISC       0.78      0.87      0.82       702
B-ORG       0.92      0.92      0.92      1661
B-PER       0.98      0.97      0.98      1617

micro avg       0.91      0.93      0.92      8112
macro avg       0.86      0.91      0.88      8112
weighted avg       0.92      0.93      0.92      8112

processed 46435 tokens with 5648 phrases; found: 5683 phrases; correct: 5170.
accuracy:  93.33%; (non-O)
accuracy:  98.22%; precision:  90.97%; recall:  91.54%; FB1:  91.25
LOC: precision:  93.75%; recall:  91.73%; FB1:  92.73  1632
MISC: precision:  74.81%; recall:  83.33%; FB1:  78.84  782
ORG: precision:  90.05%; recall:  90.49%; FB1:  90.27  1669
PER: precision:  97.00%; recall:  95.98%; FB1:  96.49  1600

Dev:

precision    recall  f1-score   support

B-LOC       0.98      0.96      0.97      1837
I-ORG       0.95      0.92      0.93       751
I-MISC       0.77      0.90      0.83       346
I-LOC       0.96      0.93      0.95       257
I-PER       0.98      0.99      0.98      1307
B-MISC       0.87      0.93      0.90       922
B-ORG       0.96      0.94      0.95      1341
B-PER       0.99      0.99      0.99      1842

micro avg       0.95      0.96      0.96      8603
macro avg       0.93      0.95      0.94      8603
weighted avg       0.96      0.96      0.96      8603

processed 51362 tokens with 5942 phrases; found: 5958 phrases; correct: 5642.
accuracy:  95.79%; (non-O)
accuracy:  99.08%; precision:  94.70%; recall:  94.95%; FB1:  94.82
LOC: precision:  97.68%; recall:  96.19%; FB1:  96.93  1809
MISC: precision:  83.45%; recall:  89.70%; FB1:  86.46  991
ORG: precision:  94.55%; recall:  93.14%; FB1:  93.84  1321
PER: precision:  97.93%; recall:  97.67%; FB1:  97.80  1837
```