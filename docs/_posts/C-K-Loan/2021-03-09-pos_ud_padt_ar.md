---
layout: model
title: Part of Speech for Arabic
author: John Snow Labs
name: pos_ud_padt
date: 2021-03-09
tags: [part_of_speech, open_source, arabic, pos_ud_padt, ar]
task: Part of Speech Tagging
language: ar
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

- X
- VERB
- NOUN
- ADJ
- ADP
- PUNCT
- NUM
- None
- PRON
- SCONJ
- CCONJ
- DET
- PART
- ADV
- SYM
- AUX
- PROPN
- INTJ

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_padt_ar_3.0.0_3.0_1615292251530.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos_tagger = PerceptronModel.pretrained("pos_ud_padt", "ar") \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  tokenizer,
  pos_tagger
])

example = spark.createDataFrame([['مرحبا من جون سنو مختبرات! ']], ["text"])

result = pipeline.fit(example).transform(example)


```
```scala

val documentAssembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentenceDetector = SentenceDetector()
        .setInputCols("document")
        .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val posTagger = PerceptronModel.pretrained("pos_ud_padt", "ar")
        .setInputCols("sentence", "token")
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, posTagger))

val data = Seq("مرحبا من جون سنو مختبرات! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""مرحبا من جون سنو مختبرات! ""]
token_df = nlu.load('ar.pos').predict(text)
token_df
    
```
</div>

## Results

```bash
     token    pos
                 
0    مرحبا   NOUN
1       من    ADP
2      جون      X
3      سنو      X
4  مختبرات   NOUN
5        !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_padt|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|ar|
