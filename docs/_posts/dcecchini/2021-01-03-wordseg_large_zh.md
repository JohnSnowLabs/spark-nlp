---
layout: model
title: Chinese Word Segmentation
author: John Snow Labs
name: wordseg_large
date: 2021-01-03
task: Word Segmentation
language: zh
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, word_segmentation, zh, cn]
supported: true
annotator: WordSegmenterModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

WordSegmenterModel (WSM) is based on maximum entropy probability model to detect word boundaries in Chinese text. Chinese text is written without white space between the words, and a computer-based application cannot know _a priori_ which sequence of ideograms form a word. In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.

In this model, we created a curated large data set obtained from Chinese Treebank, Weibo, and SIGHAM 2005 data sets, and trained the neural network model as described in a research paper (Xue, Nianwen. "Chinese word segmentation as character tagging." International Journal of Computational Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese Language Processing. 2003.).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_large_zh_2.7.0_2.4_1609681406666.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wordseg_large_zh_2.7.0_2.4_1609681406666.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
word_segmenter = WordSegmenterModel.load("WORDSEG_LARGE_CN")\
        .setInputCols("document")\
        .setOutputCol("token")\
pipeline = Pipeline(stages=[document_assembler, word_segmenter])
ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['然而，这样的处理也衍生了一些问题。']], ["text"])
result = ws_model.transform(example)
```

```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")
        .setInputCols("document")
        .setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val data = Seq("然而，这样的处理也衍生了一些问题。").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""然而，这样的处理也衍生了一些问题。"""]
token_df = nlu.load('zh.segment_words.large').predict(text, output_level='token')
token_df
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
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[token]|
|Language:|zh|

## Data Source

cn_wordseg_large_train.chartag

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
