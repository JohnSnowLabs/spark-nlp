---
layout: model
title: Part of Speech for Portuguese
author: John Snow Labs
name: pos_ud_bosque
date: 2021-03-08
tags: [part_of_speech, open_source, portuguese, pos_ud_bosque, pt]
task: Part of Speech Tagging
language: pt
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

- DET
- NOUN
- ADJ
- PROPN
- AUX
- ADP
- PUNCT
- NUM
- ADV
- VERB
- PRON
- CCONJ
- X
- SCONJ
- INTJ
- PART
- SYM

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_bosque_pt_3.0.0_3.0_1615230225146.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

sentence_detector = SentenceDetector()
  .setInputCols(["document"])
  .setOutputCol("sentence")

pos = PerceptronModel.pretrained("pos_ud_bosque", "pt")
  .setInputCols(["document", "token"])
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  posTagger
])

example = spark.createDataFrame(pd.DataFrame({'text': ["Olá de John Snow Labs! "]}))

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_bosque", "pt")
        .setInputCols(Array("document", "token"))
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val result = pipeline.fit(Seq.empty["Olá de John Snow Labs! "].toDS.toDF("text")).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""Olá de John Snow Labs! ""]
token_df = nlu.load('pt.pos').predict(text)
token_df
    
```
</div>

## Results

```bash
  token    pos
              
0   Olá  PROPN
1    de    ADP
2  John  PROPN
3  Snow  PROPN
4  Labs  PROPN
5     !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_bosque|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|pt|