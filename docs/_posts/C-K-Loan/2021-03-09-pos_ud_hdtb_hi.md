---
layout: model
title: Part of Speech for Hindi
author: John Snow Labs
name: pos_ud_hdtb
date: 2021-03-09
tags: [part_of_speech, open_source, hindi, pos_ud_hdtb, hi]
task: Part of Speech Tagging
language: hi
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
- PROPN
- ADP
- ADV
- ADJ
- NOUN
- NUM
- AUX
- PUNCT
- PRON
- VERB
- CCONJ
- PART
- SCONJ
- X
- INTJ

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_hdtb_hi_3.0.0_3.0_1615292181587.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_hdtb_hi_3.0.0_3.0_1615292181587.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

pos = PerceptronModel.pretrained("pos_ud_hdtb", "hi") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
posTagger
])

example = spark.createDataFrame([['जॉन स्नो लैब्स से नमस्ते! ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_hdtb", "hi")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("जॉन स्नो लैब्स से नमस्ते! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""जॉन स्नो लैब्स से नमस्ते! ""]
token_df = nlu.load('hi.pos').predict(text)
token_df

```
</div>

## Results

```bash
token    pos

0     जॉन  PROPN
1    स्नो  PROPN
2   लैब्स  PROPN
3      से    ADP
4  नमस्ते   NOUN
5       !   VERB
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_hdtb|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|hi|