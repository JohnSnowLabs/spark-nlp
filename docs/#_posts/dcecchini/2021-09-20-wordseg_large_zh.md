---
layout: model
title: Chinese Word Segmentation
author: John Snow Labs
name: wordseg_large
date: 2021-09-20
tags: [cn, zh, word_segmentation, open_source]
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

WordSegmenterModel (WSM) is based on a maximum entropy probability model to detect word boundaries in Chinese text. Chinese text is written without white space between the words, and a computer-based application cannot know a priori which sequence of ideograms form a word. Many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as an initial step.

Reference:

> Xue, Nianwen. “Chinese word segmentation as character tagging.” International Journal of Computational Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese Language Processing. 2003.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_large_zh_3.0.0_3.0_1632144130387.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")\
    .setInputCols("document")\
    .setOutputCol("token")
    
pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    word_segmenter
])

ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

example = spark.createDataFrame([['然而，这样的处理也衍生了一些问题。']], ["text"])

result = ws_model.transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

val tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")
    .setInputCols("document")
    .setOutputCol("token")
    
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector,tokenizer, word_segmenter))

val data = Seq("然而，这样的处理也衍生了一些问题。").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
pipe = nlu.load('zh.segment_words.large')

zh_data = ["然而，这样的处理也衍生了一些问题。"]

df = pipe.predict(zh_data , output_level='token')
df
```
</div>

## Results

```bash
+----------------------------------+--------------------------------------------------------+
|text                              |result                                                  |
+----------------------------------+--------------------------------------------------------+
|然而，这样的处理也衍生了一些问题。|[然而, ，, 这样, 的, 处理, 也, 衍生, 了, 一些, 问题, 。]|
+----------------------------------+--------------------------------------------------------+
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
|Output Labels:|[token]|
|Language:|zh|

## Data Source

We created a curated large data set obtained from Chinese Treebank, Weibo, and SIGHAM 2005 data sets.

## Benchmarking

```bash
| Model         | precision    | recall       | f1-score     |
|---------------|--------------|--------------|--------------|
| WORSEG_CTB    |      0,6453  |      0,6341  |      0,6397  |
| WORDSEG_WEIBO |      0,5454  |      0,5655  |      0,5553  |
| WORDSEG_MSRA   |      0,5984  |      0,6088  |      0,6035  |
| WORDSEG_PKU   |      0,6094  |      0,6321  |      0,6206  |
| WORDSEG_LARGE |      0,6326  |      0,6269  |      0,6297  |
```