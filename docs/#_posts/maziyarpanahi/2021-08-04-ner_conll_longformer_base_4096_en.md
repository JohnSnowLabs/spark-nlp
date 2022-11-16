---
layout: model
title: Named Entity Recognition - CoNLL03 Longformer (ner_conll_longformer_base_4096)
author: John Snow Labs
name: ner_conll_longformer_base_4096
date: 2021-08-04
tags: [ner, conll, open_source, en, english, longformer]
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

`ner_conll_longformer_base_4096` is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. It was trained on the CoNLL 2003 text corpus. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.`ner_conll_longformer_base_4096` model has trained with`longformer_base_4096` word embeddings, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `MISC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_conll_longformer_base_4096_en_3.2.0_2.4_1628081396660.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained("longformer_base_4096")\
      .setInputCols(['document', 'token'])\
      .setOutputCol("embeddings")\
      .setCaseSensitive(True)\
      .setMaxSentenceLength(4096)

ner_model = NerDLModel.pretrained('ner_conll_longformer_base_4096', 'en') \
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

val embeddings = LongformerEmbeddings.pretrained("longformer_base_4096", "en")
    .setInputCols("document", "token") 
    .setOutputCol("embeddings")
    .setCaseSensitive(true)
    .setMaxSentenceLength(4096)

val ner_model = NerDLModel.pretrained("ner_conll_longformer_base_4096", "en") 
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

ner_df = nlu.load('en.ner.ner_conll_longformer_base_4096').predict(text, output_level='token')
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_conll_longformer_base_4096|
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

       B-LOC       0.92      0.93      0.92      1668
       I-ORG       0.86      0.91      0.88       835
      I-MISC       0.68      0.69      0.69       216
       I-LOC       0.79      0.84      0.82       257
       I-PER       0.97      0.99      0.98      1156
      B-MISC       0.83      0.81      0.82       702
       B-ORG       0.91      0.89      0.90      1661
       B-PER       0.96      0.97      0.96      1617

   micro avg       0.91      0.92      0.91      8112
   macro avg       0.86      0.88      0.87      8112
weighted avg       0.91      0.92      0.91      8112

processed 46435 tokens with 5648 phrases; found: 5669 phrases; correct: 5098.
accuracy:  91.74%; (non-O)
accuracy:  98.05%; precision:  89.93%; recall:  90.26%; FB1:  90.09
              LOC: precision:  90.76%; recall:  92.51%; FB1:  91.63  1700
             MISC: precision:  80.29%; recall:  78.92%; FB1:  79.60  690
              ORG: precision:  88.45%; recall:  87.18%; FB1:  87.81  1637
              PER: precision:  94.58%; recall:  96.04%; FB1:  95.31  1642

Dev:

        precision    recall  f1-score   support

       B-LOC       0.96      0.97      0.96      1837
       I-ORG       0.95      0.92      0.93       751
      I-MISC       0.90      0.81      0.85       346
       I-LOC       0.90      0.93      0.92       257
       I-PER       0.98      0.98      0.98      1307
      B-MISC       0.92      0.88      0.90       922
       B-ORG       0.96      0.94      0.95      1341
       B-PER       0.98      0.98      0.98      1842

   micro avg       0.96      0.95      0.95      8603
   macro avg       0.94      0.93      0.93      8603
weighted avg       0.96      0.95      0.95      8603

processed 51362 tokens with 5942 phrases; found: 5943 phrases; correct: 5599.
accuracy:  94.75%; (non-O)
accuracy:  99.00%; precision:  94.21%; recall:  94.23%; FB1:  94.22
              LOC: precision:  95.08%; recall:  96.84%; FB1:  95.95  1871
             MISC: precision:  88.37%; recall:  85.68%; FB1:  87.00  894
              ORG: precision:  93.35%; recall:  92.17%; FB1:  92.76  1324
              PER: precision:  96.76%; recall:  97.39%; FB1:  97.08  1854
```