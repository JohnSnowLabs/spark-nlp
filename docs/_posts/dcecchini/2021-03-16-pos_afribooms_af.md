---
layout: model
title: Part of Speech for Afrikaans
author: John Snow Labs
name: pos_afribooms
date: 2021-03-16
tags: [af, open_source, pos]
supported: true
task: Part of Speech Tagging
language: af
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_afribooms_af_2.7.5_2.4_1615903333785.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_afribooms_af_2.7.5_2.4_1615903333785.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

pos = PerceptronModel.pretrained("pos_afribooms", "af")\
.setInputCols(["document", "token"])\
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
tokenizer,
posTagger
])

example = spark.createDataFrame([['Die kodes wat gebruik word , moet duidelik en verstaanbaar vir leerders en ouers wees .']], ["text"])
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_afribooms", "af")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer ,pos))

val data = Seq("Die kodes wat gebruik word , moet duidelik en verstaanbaar vir leerders en ouers wees .").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = [""Die kodes wat gebruik word , moet duidelik en verstaanbaar vir leerders en ouers wees .""]
token_df = nlu.load('af.pos.afribooms').predict(text)
token_df
```
</div>

## Results

```bash

+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
|text                                                                                   |result                                                                                       |
+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
|Die kodes wat gebruik word , moet duidelik en verstaanbaar vir leerders en ouers wees .|[DET, NOUN, PRON, VERB, AUX, PUNCT, AUX, ADJ, CCONJ, ADJ, ADP, NOUN, CCONJ, NOUN, AUX, PUNCT]|
+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_afribooms|
|Compatibility:|Spark NLP 2.7.5+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|af|

## Data Source

The model was trained on the [Universal Dependencies](https://www.universaldependencies.org) data set.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.60      | 0.67   | 0.63     | 665     |
| ADP          | 0.76      | 0.78   | 0.77     | 1299    |
| ADV          | 0.74      | 0.69   | 0.72     | 523     |
| AUX          | 0.85      | 0.83   | 0.84     | 663     |
| CCONJ        | 0.71      | 0.71   | 0.71     | 380     |
| DET          | 0.83      | 0.70   | 0.76     | 1014    |
| NOUN         | 0.69      | 0.72   | 0.71     | 2025    |
| NUM          | 0.76      | 0.76   | 0.76     | 42      |
| PART         | 0.67      | 0.68   | 0.68     | 322     |
| PRON         | 0.87      | 0.87   | 0.87     | 794     |
| PROPN        | 0.82      | 0.73   | 0.77     | 156     |
| PUNCT        | 0.68      | 0.70   | 0.69     | 877     |
| SCONJ        | 0.85      | 0.85   | 0.85     | 210     |
| SYM          | 0.87      | 0.88   | 0.87     | 142     |
| VERB         | 0.69      | 0.72   | 0.70     | 889     |
| X            | 0.35      | 0.14   | 0.20     | 64      |
| accuracy     |           |        | 0.74     | 10065   |
| macro avg    | 0.73      | 0.72   | 0.72     | 10065   |
| weighted avg | 0.74      | 0.74   | 0.74     | 10065   |
```