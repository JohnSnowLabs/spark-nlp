---
layout: model
title: Part of Speech for Norwegian Nynorsk
author: John Snow Labs
name: pos_ud_nynorsk
date: 2021-03-09
tags: [part_of_speech, open_source, norwegian_nynorsk, pos_ud_nynorsk, nn]
task: Part of Speech Tagging
language: nn
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

- DET
- NOUN
- ADP
- PUNCT
- CCONJ
- PRON
- VERB
- PROPN
- AUX
- ADJ
- ADV
- SCONJ
- PART
- INTJ
- NUM
- X
- SYM

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_nynorsk_nn_3.0.0_3.0_1615292123096.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos = PerceptronModel.pretrained("pos_ud_nynorsk", "nn") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
posTagger
])

example = spark.createDataFrame([['Hello from John Snow Labs!']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_nynorsk", "nn")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("Hello from John Snow Labs!").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs!""]
token_df = nlu.load('nn.pos.ud_nynorsk').predict(text)
token_df

```
</div>

## Results

```bash
token    pos

0  Hello  PROPN
1   from   NOUN
2   John   NOUN
3   Snow  PROPN
4   Labs  PROPN
5      !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_nynorsk|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|nn|