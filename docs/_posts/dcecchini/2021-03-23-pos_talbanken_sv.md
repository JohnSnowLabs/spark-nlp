---
layout: model
title: Part of Speech for Swedish
author: John Snow Labs
name: pos_talbanken
date: 2021-03-23
tags: [sv, open_source]
supported: true
task: Part of Speech Tagging
language: sv
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

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_talbanken_sv_2.7.5_2.4_1616511099635.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos = PerceptronModel.pretrained("pos_talbanken", "sv")\
.setInputCols(["document", "token"])\
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
tokenizer,
posTagger
])

example = spark.createDataFrame([["' Medicinsk bildtolk ' också skall fungera som hjälpmedel för läkaren att klarlägga sjukdomsbilden utan att patienten behöver säga ett ord ."]], ["text"])
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

val pos = PerceptronModel.pretrained("pos_talbanken", "sv")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector,tokenizer , pos))

val data = Seq(" Medicinsk bildtolk " också skall fungera som hjälpmedel för läkaren att klarlägga sjukdomsbilden utan att patienten behöver säga ett ord .").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = [""' Medicinsk bildtolk ' också skall fungera som hjälpmedel för läkaren att klarlägga sjukdomsbilden utan att patienten behöver säga ett ord .""]
token_df = nlu.load('sv.pos.talbanken').predict(text)
token_df
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|text                                                                                                                                         |result                                                                                                                            |
+---------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|' Medicinsk bildtolk ' också skall fungera som hjälpmedel för läkaren att klarlägga sjukdomsbilden utan att patienten behöver säga ett ord . |[PUNCT, ADJ, NOUN, PUNCT, ADV, AUX, VERB, SCONJ, NOUN, ADP, NOUN, PART, VERB, NOUN, ADP, SCONJ, NOUN, AUX, VERB, DET, NOUN, PUNCT]|
+---------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_talbanken|
|Compatibility:|Spark NLP 2.7.5+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|sv|

## Data Source

The model was trained on the [Universal Dependencies](https://www.universaldependencies.org) data set.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.88      | 0.89   | 0.89     | 1826    |
| ADP          | 0.96      | 0.96   | 0.96     | 2298    |
| ADV          | 0.91      | 0.87   | 0.89     | 1528    |
| AUX          | 0.91      | 0.93   | 0.92     | 1021    |
| CCONJ        | 0.95      | 0.94   | 0.94     | 791     |
| DET          | 0.92      | 0.95   | 0.93     | 1015    |
| INTJ         | 1.00      | 0.33   | 0.50     | 3       |
| NOUN         | 0.94      | 0.95   | 0.95     | 4711    |
| NUM          | 0.98      | 0.96   | 0.97     | 357     |
| PART         | 0.93      | 0.94   | 0.94     | 406     |
| PRON         | 0.94      | 0.91   | 0.92     | 1449    |
| PROPN        | 0.88      | 0.83   | 0.85     | 243     |
| PUNCT        | 0.97      | 0.98   | 0.98     | 2104    |
| SCONJ        | 0.86      | 0.82   | 0.84     | 491     |
| SYM          | 0.50      | 1.00   | 0.67     | 1       |
| VERB         | 0.90      | 0.90   | 0.90     | 2142    |
| accuracy     |           |        | 0.93     | 20386   |
| macro avg    | 0.90      | 0.89   | 0.88     | 20386   |
| weighted avg | 0.93      | 0.93   | 0.93     | 20386   |
```