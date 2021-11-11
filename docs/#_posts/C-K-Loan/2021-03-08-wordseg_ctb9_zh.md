---
layout: model
title: Word Segmenter for Chinese
author: John Snow Labs
name: wordseg_ctb9
date: 2021-03-08
tags: [word_segmentation, open_source, chinese, wordseg_ctb9, zh]
task: Word Segmentation
language: zh
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[WordSegmenterModel-WSM](https://en.wikipedia.org/wiki/Text_segmentation) is based on maximum entropy probability model to detect word boundaries in Chinese text.
            Chinese text is written without white space between the words, and a computer-based application cannot know a priori which sequence of ideograms form a word.
            In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_ctb9_zh_3.0.0_3.0_1615225768619.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

word_segmenter = WordSegmenterModel.pretrained("wordseg_ctb9", "zh")        .setInputCols(["sentence"])        .setOutputCol("token")
pipeline = Pipeline(stages=[document_assembler, word_segmenter])
ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['从John Snow Labs你好！ ']], ["text"])
result = ws_model.transform(example)

```
```scala

val word_segmenter = WordSegmenterModel.pretrained("wordseg_ctb9", "zh")
        .setInputCols(Array("sentence"))
        .setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val data = Seq("从John Snow Labs你好！ ").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""从John Snow Labs你好！ ""]
token_df = nlu.load('zh.segment_words.ctb9').predict(text)
token_df
    
```
</div>

## Results

```bash

0        从
1        J
2        o
3        h
4        n
5        S
6        n
7        o
8        w
9     Labs
10       你
11       好
12       ！
Name: token, dtype: object
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_ctb9|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[words_segmented]|
|Language:|zh|