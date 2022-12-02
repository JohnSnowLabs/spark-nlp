---
layout: model
title: Named Entity Recognition - CoNLL03 XLNet Base (ner_conll_xlnet_base_cased)
author: John Snow Labs
name: ner_conll_xlnet_base_cased
date: 2021-08-31
tags: [ner, english, en, conll, xlnet, base, open_source]
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

`ner_conll_xlnet_base_cased` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. `ner_conll_xlnet_base_cased` model is trained with`xlnet_base_cased` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_xlnet_base_cased_en_3.2.2_2.4_1630419499488.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = XlnetEmbeddings\
.pretrained('xlnet_base_cased', 'en')\
.setInputCols(["token", "document"])\
.setOutputCol("embeddings")

ner_model = NerDLModel.pretrained('ner_conll_xlnet_base_cased', 'en') \
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

val embeddings = XlnetEmbeddings.pretrained("xlnet_base_cased", "en")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

val ner_model = NerDLModel.pretrained("ner_conll_xlnet_base_cased", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_xlnet_base_cased').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_xlnet_base_cased|
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

B-LOC       0.87      0.89      0.88      1668
I-ORG       0.81      0.82      0.81       835
I-MISC       0.51      0.61      0.55       216
I-LOC       0.71      0.82      0.76       257
I-PER       0.93      0.99      0.96      1156
B-MISC       0.76      0.76      0.76       702
B-ORG       0.86      0.80      0.83      1661
B-PER       0.90      0.94      0.92      1617

micro avg       0.85      0.86      0.86      8112
macro avg       0.79      0.83      0.81      8112
weighted avg       0.85      0.86      0.86      8112


processed 46435 tokens with 5648 phrases; found: 5630 phrases; correct: 4782.
accuracy:  86.49%; (non-O)
accuracy:  96.74%; precision:  84.94%; recall:  84.67%; FB1:  84.80
LOC: precision:  86.43%; recall:  88.19%; FB1:  87.30  1702
MISC: precision:  72.02%; recall:  72.22%; FB1:  72.12  704
ORG: precision:  84.40%; recall:  77.84%; FB1:  80.99  1532
PER: precision:  89.30%; recall:  93.44%; FB1:  91.33  1692


Dev:

precision    recall  f1-score   support

B-LOC       0.91      0.93      0.92      1837
I-ORG       0.89      0.80      0.84       751
I-MISC       0.82      0.68      0.74       346
I-LOC       0.76      0.87      0.81       257
I-PER       0.95      0.97      0.96      1307
B-MISC       0.86      0.81      0.84       922
B-ORG       0.92      0.83      0.87      1341
B-PER       0.92      0.96      0.94      1842

micro avg       0.90      0.89      0.90      8603
macro avg       0.88      0.86      0.87      8603
weighted avg       0.90      0.89      0.90      8603


processed 51362 tokens with 5942 phrases; found: 5893 phrases; correct: 5254.
accuracy:  89.14%; (non-O)
accuracy:  97.83%; precision:  89.16%; recall:  88.42%; FB1:  88.79
LOC: precision:  89.97%; recall:  92.27%; FB1:  91.10  1884
MISC: precision:  82.49%; recall:  78.20%; FB1:  80.29  874
ORG: precision:  88.99%; recall:  80.76%; FB1:  84.68  1217
PER: precision:  91.50%; recall:  95.28%; FB1:  93.35  1918
```