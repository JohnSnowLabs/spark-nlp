---
layout: model
title: Part of Speech for Afrikaans
author: John Snow Labs
name: pos_afribooms
date: 2021-04-06
tags: [pos, open_source, af]
task: Part of Speech Tagging
language: af
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_afribooms_af_3.0.0_3.0_1617749039095.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.setInputCols(["document"])
	.setOutputCol("sentence")

val tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_afribooms", "af")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector,tokenizer ,pos))

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
|Compatibility:|Spark NLP 3.0.0+|
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
| ADJ          | 0.59      | 0.65   | 0.62     | 665     |
| ADP          | 0.76      | 0.79   | 0.77     | 1299    |
| ADV          | 0.72      | 0.69   | 0.71     | 523     |
| AUX          | 0.86      | 0.83   | 0.84     | 663     |
| CCONJ        | 0.70      | 0.71   | 0.70     | 380     |
| DET          | 0.84      | 0.70   | 0.76     | 1014    |
| NOUN         | 0.68      | 0.72   | 0.70     | 2025    |
| NUM          | 0.89      | 0.81   | 0.85     | 42      |
| PART         | 0.67      | 0.68   | 0.67     | 322     |
| PRON         | 0.87      | 0.87   | 0.87     | 794     |
| PROPN        | 0.87      | 0.67   | 0.75     | 156     |
| PUNCT        | 0.68      | 0.70   | 0.69     | 877     |
| SCONJ        | 0.82      | 0.85   | 0.83     | 210     |
| SYM          | 0.86      | 0.88   | 0.87     | 142     |
| VERB         | 0.69      | 0.71   | 0.70     | 889     |
| X            | 0.24      | 0.14   | 0.18     | 64      |
| accuracy     |           |        | 0.73     | 10065   |
| macro avg    | 0.73      | 0.71   | 0.72     | 10065   |
| weighted avg | 0.74      | 0.73   | 0.73     | 10065   |
```