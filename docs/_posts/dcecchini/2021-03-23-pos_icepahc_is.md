---
layout: model
title: Part of Speech for Icelandic
author: John Snow Labs
name: pos_icepahc
date: 2021-03-23
tags: [pos, open_source, is]
supported: true
task: Part of Speech Tagging
language: is
edition: Spark NLP 2.7.5
spark_version: 2.4
annotator: PerceptronModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A [Part of Speech](https://en.wikipedia.org/wiki/Part_of_speech) classifier predicts a grammatical label for every token in the input text. Implemented with an `averaged perceptron` architecture.

## Predicted Entities

- ADJ
- ADP
- ADV
- AUX
- CCONJ
- DET
- NOUN
- NUM
- PART
- PRON
- PROPN
- PUNCT
- VERB
- X

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_icepahc_is_2.7.5_2.4_1616509019245.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_icepahc_is_2.7.5_2.4_1616509019245.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentence_detector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

pos = PerceptronModel.pretrained("pos_icepahc", "is")\
  .setInputCols(["document", "token"])\
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  tokenizer,
  posTagger
])

example = spark.createDataFrame([['Númerið blikkaði á skjánum eins og einmana vekjaraklukka um nótt á níundu hæð í gamalli blokk í austurbæ Reykjavíkur .']], ["text"])
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
	.setOutputCol("sentence")

val tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_icepahc", "is")
        .setInputCols(Array("document", "token"))
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector,tokenizer, pos))

val data = Seq("Númerið blikkaði á skjánum eins og einmana vekjaraklukka um nótt á níundu hæð í gamalli blokk í austurbæ Reykjavíkur .").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = [""Númerið blikkaði á skjánum eins og einmana vekjaraklukka um nótt á níundu hæð í gamalli blokk í austurbæ Reykjavíkur .""]
token_df = nlu.load('is.pos.icepahc').predict(text)
token_df
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|text                                                                                                                  |result                                                                                                           |
+----------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|Númerið blikkaði á skjánum eins og einmana vekjaraklukka um nótt á níundu hæð í gamalli blokk í austurbæ Reykjavíkur .|[NOUN, VERB, ADP, NOUN, ADV, ADP, ADJ, NOUN, ADP, NOUN, ADP, ADJ, NOUN, ADP, ADJ, NOUN, ADP, PROPN, PROPN, PUNCT]|
+----------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_icepahc|
|Compatibility:|Spark NLP 2.7.5+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|is|

## Data Source

The model was trained on the [Universal Dependencies](https://www.universaldependencies.org) data set.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.81      | 0.74   | 0.78     | 5906    |
| ADP          | 0.95      | 0.96   | 0.96     | 15548   |
| ADV          | 0.90      | 0.90   | 0.90     | 10631   |
| AUX          | 0.92      | 0.93   | 0.92     | 7416    |
| CCONJ        | 0.96      | 0.97   | 0.96     | 8437    |
| DET          | 0.89      | 0.87   | 0.88     | 7476    |
| INTJ         | 0.95      | 0.77   | 0.85     | 131     |
| NOUN         | 0.90      | 0.92   | 0.91     | 20726   |
| NUM          | 0.75      | 0.83   | 0.79     | 655     |
| PART         | 0.96      | 0.96   | 0.96     | 1703    |
| PRON         | 0.94      | 0.96   | 0.95     | 16852   |
| PROPN        | 0.89      | 0.89   | 0.89     | 4444    |
| PUNCT        | 0.98      | 0.98   | 0.98     | 16434   |
| SCONJ        | 0.94      | 0.94   | 0.94     | 5663    |
| VERB         | 0.92      | 0.90   | 0.91     | 17329   |
| X            | 0.60      | 0.30   | 0.40     | 346     |
| accuracy     |           |        | 0.92     | 139697  |
| macro avg    | 0.89      | 0.86   | 0.87     | 139697  |
| weighted avg | 0.92      | 0.92   | 0.92     | 139697  |
```