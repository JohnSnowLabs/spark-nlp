---
layout: model
title: Named Entity Recognition - CoNLL03 ELMO Base (ner_conll_elmo)
author: John Snow Labs
name: ner_conll_elmo
date: 2021-08-31
tags: [ner, english, en, elmo, open_source, conll]
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

`ner_conll_elmo` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. `ner_conll_elmo` model is trained with`elmo` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_elmo_en_3.2.2_2.4_1630419727678.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_conll_elmo_en_3.2.2_2.4_1630419727678.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = ElmoEmbeddings\
      .pretrained('elmo', 'en')\
      .setInputCols(["token", "document"])\
      .setOutputCol("embeddings")\
      .setPoolingLayer("elmo")

ner_model = NerDLModel.pretrained('ner_conll_elmo', 'en') \
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

val embeddings = ElmoEmbeddings.pretrained("elmo", "en")
    .setInputCols("document", "token") 
    .setOutputCol("embeddings")
    .setPoolingLayer("elmo")

val ner_model = NerDLModel.pretrained("ner_conll_elmo", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_elmo').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_elmo|
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

       B-LOC       0.93      0.93      0.93      1668
       I-ORG       0.87      0.92      0.90       835
      I-MISC       0.68      0.71      0.69       216
       I-LOC       0.84      0.91      0.87       257
       I-PER       0.98      0.99      0.99      1156
      B-MISC       0.84      0.83      0.83       702
       B-ORG       0.89      0.92      0.91      1661
       B-PER       0.97      0.96      0.97      1617

   micro avg       0.91      0.93      0.92      8112
   macro avg       0.87      0.90      0.89      8112
weighted avg       0.91      0.93      0.92      8112


processed 46435 tokens with 5648 phrases; found: 5686 phrases; correct: 5176.
accuracy:  92.84%; (non-O)
accuracy:  98.17%; precision:  91.03%; recall:  91.64%; FB1:  91.34
              LOC: precision:  92.32%; recall:  92.93%; FB1:  92.62  1679
             MISC: precision:  82.00%; recall:  80.48%; FB1:  81.24  689
              ORG: precision:  87.86%; recall:  90.67%; FB1:  89.24  1714
              PER: precision:  96.95%; recall:  96.17%; FB1:  96.55  1604


Dev:

              precision    recall  f1-score   support

       B-LOC       0.97      0.97      0.97      1837
       I-ORG       0.93      0.94      0.93       751
      I-MISC       0.93      0.85      0.89       346
       I-LOC       0.92      0.93      0.93       257
       I-PER       0.98      0.98      0.98      1307
      B-MISC       0.94      0.90      0.92       922
       B-ORG       0.93      0.95      0.94      1341
       B-PER       0.97      0.98      0.98      1842

   micro avg       0.96      0.95      0.96      8603
   macro avg       0.95      0.94      0.94      8603
weighted avg       0.96      0.95      0.95      8603


processed 51362 tokens with 5942 phrases; found: 5944 phrases; correct: 5641.
accuracy:  95.46%; (non-O)
accuracy:  99.07%; precision:  94.90%; recall:  94.93%; FB1:  94.92
              LOC: precision:  96.57%; recall:  96.68%; FB1:  96.63  1839
             MISC: precision:  92.18%; recall:  88.18%; FB1:  90.13  882
              ORG: precision:  91.92%; recall:  93.36%; FB1:  92.64  1362
              PER: precision:  96.72%; recall:  97.72%; FB1:  97.22  1861
```