---
layout: model
title: Part of Speech for Latin
author: John Snow Labs
name: pos_ud_llct
date: 2021-03-09
tags: [part_of_speech, open_source, latin, pos_ud_llct, la]
task: Part of Speech Tagging
language: la
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

- PUNCT
- ADP
- PROPN
- NOUN
- VERB
- DET
- CCONJ
- PRON
- ADJ
- NUM
- AUX
- SCONJ
- ADV
- PART
- X

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_llct_la_3.0.0_3.0_1615292206384.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos = PerceptronModel.pretrained("pos_ud_llct", "la") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  posTagger
])

example = spark.createDataFrame([['Aequaliter Nubila Labs Ioannes de salve ! ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_llct", "la")
        .setInputCols(Array("document", "token"))
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("Aequaliter Nubila Labs Ioannes de salve ! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""Aequaliter Nubila Labs Ioannes de salve ! ""]
token_df = nlu.load('la.pos').predict(text)
token_df
    
```
</div>

## Results

```bash
        token    pos
                    
0  Aequaliter  PROPN
1      Nubila  PROPN
2        Labs    ADJ
3     Ioannes   NOUN
4          de    ADP
5       salve   NOUN
6           !  PROPN
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_llct|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|la|