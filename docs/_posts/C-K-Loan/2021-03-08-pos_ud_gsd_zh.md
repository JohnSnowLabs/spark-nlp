---
layout: model
title: Part of Speech for Chinese
author: John Snow Labs
name: pos_ud_gsd
date: 2021-03-08
tags: [part_of_speech, open_source, chinese, pos_ud_gsd, zh]
task: Part of Speech Tagging
language: zh
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

- VERB
- ADJ
- PUNCT
- ADV
- NUM
- NOUN
- PRON
- PART
- X
- ADP
- DET
- CONJ
- PROPN
- AUX
- SYM

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_zh_3.0.0_3.0_1615230264489.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos = PerceptronModel.pretrained("pos_ud_gsd", "zh") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  posTagger
])

example = spark.createDataFrame([['从John Snow Labs你好！ ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_gsd", "zh")
        .setInputCols(Array("document", "token"))
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("从John Snow Labs你好！ ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""从John Snow Labs你好！ ""]
token_df = nlu.load('zh.pos.ud_gsd').predict(text)
token_df
    
```
</div>

## Results

```bash
          token   pos
                     
0             从   ADP
1  JohnSnowLabs     X
2             你  PRON
3             好   ADV
4             ！  VERB
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
|Language:|zh|