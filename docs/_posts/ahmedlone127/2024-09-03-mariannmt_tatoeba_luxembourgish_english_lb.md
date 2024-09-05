---
layout: model
title: Letzeburgesch, Luxembourgish mariannmt_tatoeba_luxembourgish_english MarianTransformer from mbarnig
author: John Snow Labs
name: mariannmt_tatoeba_luxembourgish_english
date: 2024-09-03
tags: [lb, open_source, onnx, translation, marian]
task: Translation
language: lb
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: MarianTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mariannmt_tatoeba_luxembourgish_english` is a Letzeburgesch, Luxembourgish model originally trained by mbarnig.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mariannmt_tatoeba_luxembourgish_english_lb_5.5.0_3.0_1725404608318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mariannmt_tatoeba_luxembourgish_english_lb_5.5.0_3.0_1725404608318.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
      .setInputCols(["document"]) \
      .setOutputCol("translation")

marian = MarianTransformer.pretrained("mariannmt_tatoeba_luxembourgish_english","lb") \
      .setInputCols(["sentence"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, sentenceDL, marian])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val marian = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val embeddings = MarianTransformer.pretrained("mariannmt_tatoeba_luxembourgish_english","lb") 
    .setInputCols(Array("sentence")) 
    .setOutputCol("translation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDL, marian))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mariannmt_tatoeba_luxembourgish_english|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentences]|
|Output Labels:|[translation]|
|Language:|lb|
|Size:|284.9 MB|

## References

https://huggingface.co/mbarnig/marianNMT-tatoeba-lb-en