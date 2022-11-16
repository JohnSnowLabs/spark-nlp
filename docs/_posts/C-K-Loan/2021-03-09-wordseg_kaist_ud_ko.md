---
layout: model
title: Word Segmenter for Korean
author: John Snow Labs
name: wordseg_kaist_ud
date: 2021-03-09
tags: [word_segmentation, open_source, korean, wordseg_kaist_ud, ko]
task: Word Segmentation
language: ko
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
annotator: WordSegmenterModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[WordSegmenterModel-WSM](https://en.wikipedia.org/wiki/Text_segmentation) is based on maximum entropy probability model to detect word boundaries in Korean text.
Korean text is written without white space between the words, and a computer-based application cannot know a priori which sequence of ideograms form a word.
In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_kaist_ud_ko_3.0.0_3.0_1615292316292.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")        .setInputCols(["sentence"])        .setOutputCol("token")
pipeline = Pipeline(stages=[document_assembler, word_segmenter])
ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['John Snow Labs에서 안녕하세요! ']], ["text"])
result = ws_model.transform(example)

```
```scala

val word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")
.setInputCols(Array("sentence"))
.setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val data = Seq("John Snow Labs에서 안녕하세요! ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""John Snow Labs에서 안녕하세요! ""]
token_df = nlu.load('ko.segment_words').predict(text)
token_df

```
</div>

## Results

```bash

0       J
1       o
2       h
3       n
4       S
5       n
6       o
7       w
8       L
9       a
10      b
11      s
12      에
13      서
14      안
15      녕
16    하세요
17      !
Name: token, dtype: object
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_kaist_ud|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[words_segmented]|
|Language:|ko|