---
layout: model
title: Word Segmenter for Chinese
author: John Snow Labs
name: wordseg_msra
date: 2021-03-09
tags: [word_segmentation, open_source, chinese, wordseg_msra, zh]
task: Word Segmentation
language: zh
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
annotator: WordSegmenterModel
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_msra_zh_3.0.0_3.0_1615292321709.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wordseg_msra_zh_3.0.0_3.0_1615292321709.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

word_segmenter = WordSegmenterModel.pretrained("wordseg_msra", "zh")        .setInputCols(["sentence"])        .setOutputCol("token")
pipeline = Pipeline(stages=[document_assembler, word_segmenter])
ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['从John Snow Labs你好！ ']], ["text"])
result = ws_model.transform(example)

```
```scala

val word_segmenter = WordSegmenterModel.pretrained("wordseg_msra", "zh")
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
token_df = nlu.load('zh.segment_words.msra').predict(text)
token_df
    
```
</div>

## Results

```bash

0     从
1     J
2     o
3     h
4     n
5     S
6     n
7     o
8     w
9     L
10    a
11    b
12    s
13    你
14    好
15    ！
Name: token, dtype: object
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_msra|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[words_segmented]|
|Language:|zh|