---
layout: model
title: Word Segmenter for Chinese
author: John Snow Labs
name: wordseg_large
date: 2021-03-08
tags: [word_segmentation, open_source, chinese, wordseg_large, zh]
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

WordSegmenterModel (WSM) is based on maximum entropy probability model to detect word boundaries in Chinese text.
            Chinese text is written without white space between the words, and a computer-based application cannot know a priori which sequence of ideograms form a word.
            In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_large_zh_3.0.0_3.0_1615211653426.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")        .setInputCols(["sentence"])        .setOutputCol("token")
pipeline = Pipeline(stages=[document_assembler, word_segmenter])
ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame(pd.DataFrame({'text': [""从John Snow Labs你好！ ""]}))
result = ws_model.transform(example)

```
```scala

val word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")
        .setInputCols(Array("sentence"))
        .setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val result = pipeline.fit(Seq.empty["从John Snow Labs你好！ "].toDS.toDF("text")).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""从John Snow Labs你好！ ""]
token_df = nlu.load('zh.segment_words.large').predict(text)
token_df
    
```
</div>

## Results

```bash

0               从
1    JohnSnowLabs
2               你
3               好
4               ！
Name: token, dtype: object
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_large|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[words_segmented]|
|Language:|zh|