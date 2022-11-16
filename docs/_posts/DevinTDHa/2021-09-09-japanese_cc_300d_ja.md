---
layout: model
title: Word Embeddings for Japanese (japanese_cc_300d)
author: John Snow Labs
name: japanese_cc_300d
date: 2021-09-09
tags: [embeddings, open_source, ja]
task: Embeddings
language: ja
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
annotator: WordEmbeddingsModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained on Common Crawl and Wikipedia using fastText. It is trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.

The model gives 300 dimensional vector outputs per token. The output vectors map words into a meaningful space where the distance between the vectors is related to semantic similarity of words.

These embeddings can be used in multiple tasks like semantic word similarity, named entity recognition, sentiment analysis, and classification.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/japanese_cc_300d_ja_3.2.2_3.0_1631192388744.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja") \
.setInputCols(["sentence"]) \
.setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("japanese_cc_300d", "ja") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
word_segmenter,
embeddings
])

data = spark.createDataFrame([["宮本茂氏は、日本の任天堂のゲームプロデューサーです。"]]).toDF("text")
model = pipeline.fit(data)
result = model.transform(data)
result.selectExpr("explode(arrays_zip(embeddings.result, embeddings.embeddings))").show()
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, WordSegmenterModel}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = new SentenceDetector()
.setInputCols("document")
.setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")
.setInputCols("sentence")
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("japanese_cc_300d", "ja")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentence,
word_segmenter,
embeddings
))

val data = Seq("宮本茂氏は、日本の任天堂のゲームプロデューサーです。").toDF("text")
val model = pipeline.fit(data)
val result = model.transform(data)
result.selectExpr("explode(arrays_zip(embeddings.result, embeddings.embeddings))").show()
```


{:.nlu-block}
```python
import nlu
nlu.load("ja.embed.glove.cc_300d").predict("""explode(arrays_zip(embeddings.result, embeddings.embeddings))""")
```

</div>

## Results

```bash
+---------------------------+
|                        col|
+---------------------------+
|     [宮本, [0.1944, 0.4...|
|      [茂, [-0.079, 0.09...|
|      [氏, [-0.1053, 0.1...|
|      [は, [0.0732, -0.0...|
|      [、, [0.0571, -0.0...|
|     [日本, [0.1844, 0.0...|
|      [の, [0.0109, -0.0...|
|     [任天, [0.0, 0.0, 0...|
|      [堂, [-0.1972, 0.0...|
|      [の, [0.0109, -0.0...|
|    [ゲーム, [0.013, 0.0...|
|[プロデューサー, [-0.010...|
|     [です, [0.0036, -0....|
|      [。, [0.069, -0.01...|
+---------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|japanese_cc_300d|
|Type:|embeddings|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|ja|
|Case sensitive:|false|
|Dimension:|300|

## Data Source

This model is imported from https://fasttext.cc/docs/en/crawl-vectors.html