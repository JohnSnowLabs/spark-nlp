---
layout: model
title: Part of Speech for Korean
author: John Snow Labs
name: pos_ud_kaist
date: 2021-03-09
tags: [part_of_speech, open_source, korean, pos_ud_kaist, ko]
task: Part of Speech Tagging
language: ko
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

- CCONJ
- ADV
- SCONJ
- DET
- NOUN
- VERB
- ADJ
- PUNCT
- AUX
- PRON
- PROPN
- NUM
- INTJ
- PART
- X
- ADP
- SYM

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_kaist_ko_3.0.0_3.0_1615292391244.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos = PerceptronModel.pretrained("pos_ud_kaist", "ko") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
posTagger
])

example = spark.createDataFrame([['John Snow Labs에서 안녕하세요! ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_kaist", "ko")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("John Snow Labs에서 안녕하세요! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""John Snow Labs에서 안녕하세요! ""]
token_df = nlu.load('ko.pos.ud_kaist').predict(text)
token_df

```
</div>

## Results

```bash
token    pos

0      J   NOUN
1      o   NOUN
2      h   NOUN
3      n  SCONJ
4      S      X
5      n      X
6      o      X
7      w      X
8      L      X
9      a      X
10     b      X
11     s      X
12     에    ADP
13     서  SCONJ
14     안    ADV
15     녕   VERB
16   하세요   VERB
17     !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_kaist|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|ko|