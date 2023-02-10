---
layout: model
title: Part of Speech for Bengali
author: John Snow Labs
name: pos_msri
date: 2021-03-09
tags: [part_of_speech, open_source, bengali, pos_msri, bn]
task: Part of Speech Tagging
language: bn
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

- NN
- SYM
- NNP
- VM
- INTF
- JJ
- QF
- CC
- NST
- PSP
- QC
- DEM
- RDP
- PRP
- NEG
- WQ
- RB
- VAUX
- UT
- XC
- RP
- QO
- BM
- NNC
- PPR
- INJ
- CL
- UNK

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_msri_bn_3.0.0_3.0_1615292420029.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_msri_bn_3.0.0_3.0_1615292420029.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols(["document"]) \
.setOutputCol("token")

posTagger = PerceptronModel.pretrained("pos_msri", "bn") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[document_assembler, tokenizer, posTagger])

example = spark.createDataFrame([['জন স্নো ল্যাবস থেকে হ্যালো! ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_msri", "bn")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val data = Seq("জন স্নো ল্যাবস থেকে হ্যালো! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""জন স্নো ল্যাবস থেকে হ্যালো! ""]
token_df = nlu.load('bn.pos').predict(text)
token_df

```
</div>

## Results

```bash
token  pos

0      জন   NN
1    স্নো   NN
2  ল্যাবস   NN
3    থেকে  PSP
4  হ্যালো   JJ
5       !  SYM
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_msri|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|bn|