---
layout: model
title: Japanese Word Segmentation
author: John Snow Labs
name: wordseg_gsd_ud
date: 2021-01-03
task: Word Segmentation
language: ja
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [word_segmentation, ja, open_source]
supported: true
annotator: WordSegmenterModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

WordSegmenterModel (WSM) is based on maximum entropy probability model to detect word boundaries in Chinese text. Chinese text is written without white space between the words, and a computer-based application cannot know _a priori_ which sequence of ideograms form a word. In many natural language processing tasks such as part-of-speech (POS) and named entity recognition (NER) require word segmentation as a initial step.


References:

- Xue, Nianwen. "Chinese word segmentation as character tagging." International Journal of Computational Linguistics & Chinese Language Processing, Volume 8, Number 1, February 2003: Special Issue on Word Formation and Chinese Language Processing. 2003.).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wordseg_gsd_ud_ja_2.7.0_2.4_1609692613721.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wordseg_gsd_ud_ja_2.7.0_2.4_1609692613721.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as part of an nlp pipeline as a substitute of the Tokenizer stage.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
word_segmenter = WordSegmenterModel.pretrained('wordseg_gsd_ud', 'ja')\
.setInputCols("document")\
.setOutputCol("token")     
pipeline = Pipeline(stages=[
document_assembler,
word_segmenter
])
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
example = spark.createDataFrame([['清代は湖北省が置かれ、そのまま現代の行政区分になっている。']], ["text"])
result = model.transform(example)
```

```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")
.setInputCols("document")
.setOutputCol("token")
val pipeline = new Pipeline().setStages(Array(document_assembler, word_segmenter))
val data = Seq("清代は湖北省が置かれ、そのまま現代の行政区分になっている。").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""清代は湖北省が置かれ、そのまま現代の行政区分になっている。"""]
token_df = nlu.load('ja.segment_words').predict(text, output_level='token')
token_df
```

</div>

## Results

```bash
+----------------------------------------------------------+------------------------------------------------------------------------------------------------+
|text                                                      |result                                                                                          |
+----------------------------------------------------------+------------------------------------------------------------------------------------------------+
|清代は湖北省が置かれ、そのまま現代の行政区分になっている。|[清代, は, 湖北, 省, が, 置か, れ, 、, その, まま, 現代, の, 行政, 区分, に, なっ, て, いる, 。]|
+----------------------------------------------------------+------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wordseg_gsd_ud|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[token]|
|Language:|ja|

## Data Source

We trained this model on the the [Universal Dependenicies](universaldependencies.org) data set from Google (GSD-UD).

> Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
| Model         | precision    | recall       | f1-score     |
|---------------|--------------|--------------|--------------|
| JA_UD-GSD     |      0,7687  |      0,8048  |      0,7863  |
```
