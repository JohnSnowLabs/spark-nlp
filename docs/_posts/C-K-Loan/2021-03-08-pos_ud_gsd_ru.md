---
layout: model
title: Part of Speech for Russian
author: John Snow Labs
name: pos_ud_gsd
date: 2021-03-08
tags: [part_of_speech, open_source, russian, pos_ud_gsd, ru]
task: Part of Speech Tagging
language: ru
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A [Part of Speech](https://en.wikipedia.org/wiki/Part_of_speech) classifier predicts a grammatical label for every token in the input text. Implemented with an `averaged perceptron architecture`.

## Predicted Entities

- ADP
- NOUN
- PROPN
- CCONJ
- PUNCT
- PRON
- VERB
- DET
- ADJ
- SCONJ
- AUX
- PART
- NUM
- ADV
- SYM
- X

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_ru_3.0.0_3.0_1615230203372.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_ru_3.0.0_3.0_1615230203372.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

pos = PerceptronModel.pretrained("pos_ud_gsd", "ru") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
posTagger
])

example = spark.createDataFrame([['Здравствуйте из Джона Снежных Лабораторий! ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_gsd", "ru")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("Здравствуйте из Джона Снежных Лабораторий! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""Здравствуйте из Джона Снежных Лабораторий! ""]
token_df = nlu.load('ru.pos').predict(text)
token_df

```
</div>

## Results

```bash
token    pos

0  Здравствуйте   NOUN
1            из    ADP
2         Джона  PROPN
3       Снежных    ADJ
4   Лабораторий  PROPN
5             !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gsd|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|ru|