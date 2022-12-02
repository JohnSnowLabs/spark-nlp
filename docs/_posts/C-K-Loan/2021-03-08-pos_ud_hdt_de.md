---
layout: model
title: Part of Speech for German
author: John Snow Labs
name: pos_ud_hdt
date: 2021-03-08
tags: [part_of_speech, open_source, german, pos_ud_hdt, de]
task: Part of Speech Tagging
language: de
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
annotator: PerceptronModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A [Part of Speech](https://en.wikipedia.org/wiki/Part_of_speech) classifier predicts a grammatical label for every token in the input text. Implemented with an `averaged perceptron architecture`.

## Predicted Entities

- ADP
- DET
- ADJ
- NOUN
- VERB
- PRON
- PROPN
- X
- PUNCT
- CCONJ
- NUM
- ADV
- AUX
- SCONJ
- PART
- INTJ

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/german/pretrained-german-models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_hdt_de_3.0.0_3.0_1615230160154.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence_detector = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols("sentence") \
.setOutputCol("token")

posTagger = PerceptronModel.pretrained("pos_ud_hdt", "de") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
tokenizer,
posTagger
])

data = spark.createDataFrame([["Hallo aus John Snow Labs! "]], ["text"])

result = pipeline.fit(data).transform(data)
```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_hdt", "de")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("Hallo aus John Snow Labs! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = ["Hallo aus John Snow Labs!"]
token_df = nlu.load('de.pos').predict(text)
token_df

```
</div>

## Results

```bash
token    pos

0  Hallo   NOUN
1    aus    ADP
2   John  PROPN
3   Snow  PROPN
4   Labs  PROPN
5      !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_hdt|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|de|