---
layout: model
title: Traditional Chinese Word Segmentation
author: John Snow Labs
name: wordseg_gsd_ud_trad
date: 2021-01-25
task: Word Segmentation
language: zh
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [word_segmentation, zh, open_source]
supported: true
annotator: WordSegmenterModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

WordSegmenterModel (WSM) is based on maximum entropy probability model to detect word boundaries in Chinese text. Chinese text is written without white space between the words, and a computer-based application cannot know a priori which sequence of ideograms form a word. In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step. This model was trained on *traditional characters* in Chinese texts.

Reference:

- Xue, Nianwen. “Chinese word segmentation as character tagging.” International Journal of Computational Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese Language Processing. 2003.).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_gsd_ud_trad_zh_2.7.0_2.4_1611584735643.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline as a substitute for the Tokenizer stage.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud_trad", "zh")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")    
pipeline = Pipeline(stages=[document_assembler,sentence_detector, word_segmenter])
ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['然而，這樣的處理也衍生了一些問題。']], ["text"])
result = ws_model.transform(example)
```

```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud_trad", "zh")
        .setInputCols(Array("sentence"))
        .setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler,sentence_detector, word_segmenter))
val data = Seq("然而，這樣的處理也衍生了一些問題。").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""然而，這樣的處理也衍生了一些問題。"""]
token_df = nlu.load('zh.segment_words.gsd').predict(text)
token_df
```

</div>

## Results

```bash
+-----------------------------------------+-----------------------------------------------------------+
|text                                     | result                                                    |
+-----------------------------------------+-----------------------------------------------------------+
|然而 ， 這樣 的 處理 也 衍生 了 一些 問題 。 |[然而, ，, 這樣, 的, 處理, 也, 衍生, 了, 一些, 問題, 。]      |
+-----------------------------------------+-----------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_gsd_ud_trad|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[token]|
|Language:|zh|

## Data Source

The model was trained on the [Universal Dependencies](https://universaldependencies.org/) for Traditional Chinese annotated and converted by Google.

## Benchmarking

```bash
| precision    | recall   | f1-score   |
|--------------|----------|------------|
| 0.7392       | 0.7754   | 0.7569     |
```