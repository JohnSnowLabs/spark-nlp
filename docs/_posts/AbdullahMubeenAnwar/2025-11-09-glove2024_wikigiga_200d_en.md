---
layout: model
title: GloVe 2024 Wikipedia + Gigaword 5 (200d, uncased)
author: John Snow Labs
name: glove2024_wikigiga_200d
date: 2025-11-09
tags: [word_embedding, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 6.0.0
spark_version: 3.0
supported: true
annotator: WordEmbeddingsModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model provides 200-dimensional pre-trained word embeddings trained on the 2024 Wikipedia and Gigaword 5 corpora (11.9B tokens, 1.2M vocabulary).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/glove2024_wikigiga_200d_en_6.0.0_3.0_1762676792461.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/glove2024_wikigiga_200d_en_6.0.0_3.0_1762676792461.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings_loaded = WordEmbeddingsModel.pretrained("./glove2024_wikigiga_200d")\
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings_loaded,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a test sentence for an embedding model"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("finished_embeddings").show(truncate=False)
```
```scala
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings_loaded = WordEmbeddingsModel
  .pretrained("./glove2024_wikigiga_200d")
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings_loaded,
  embeddingsFinisher
))

val data = Seq("This is a test sentence for an embedding model").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("finished_embeddings").show(false)
```
</div>

## Results

```bash

+-------------------------------+
| embeddings                    |
+-------------------------------+
|[[-0.08433199673891068, -0.2...|
+-------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|glove2024_wikigiga_200d|
|Type:|embeddings|
|Compatibility:|Spark NLP 6.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|974.1 MB|
|Case sensitive:|false|
|Dimension:|200|