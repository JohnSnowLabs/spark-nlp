---
layout: model
title: DistilBERT base model (Cased, ONNX)
author: John Snow Labs
name: distilbert_base_cased
date: 2025-11-07
tags: [embedding, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 6.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a distilled version of the [BERT base model](https://huggingface.co/bert-base-cased). It was introduced in [this paper](https://arxiv.org/abs/1910.01108). The code for the distillation process can be found [here](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation). This model is cased: it does make a difference between english and English.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_cased_en_6.1.0_3.0_1762495659699.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_cased_en_6.1.0_3.0_1762495659699.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector, 
    tokenizer, 
    embeddings
])

data = spark.createDataFrame([["This is a test sentence for DistilBERT embeddings."]]).toDF("text")

result = nlp_pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show()
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en")
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings
))

val data = Seq("This is a test sentence for DistilBERT embeddings.").toDF("text")

val result = pipeline.fit(data).transform(data)
result.select("embeddings").show(false)
```
</div>

## Results

```bash

+--------------------+
|          embeddings|
+--------------------+
|[[0.1959381, -0.2...|
+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_cased|
|Compatibility:|Spark NLP 6.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|243.8 MB|
|Case sensitive:|true|
|Max sentence length:|512|