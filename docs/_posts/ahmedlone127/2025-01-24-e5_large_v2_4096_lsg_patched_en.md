---
layout: model
title: English e5_large_v2_4096_lsg_patched E5Embeddings from guymorganb
author: John Snow Labs
name: e5_large_v2_4096_lsg_patched
date: 2025-01-24
tags: [en, open_source, onnx, embeddings, e5]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: E5Embeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained E5Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`e5_large_v2_4096_lsg_patched` is a English model originally trained by guymorganb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_large_v2_4096_lsg_patched_en_5.5.1_3.0_1737700786186.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_large_v2_4096_lsg_patched_en_5.5.1_3.0_1737700786186.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
embeddings = E5Embeddings.pretrained("e5_large_v2_4096_lsg_patched","en") \
      .setInputCols(["document"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, embeddings])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val embeddings = E5Embeddings.pretrained("e5_large_v2_4096_lsg_patched","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|e5_large_v2_4096_lsg_patched|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[E5]|
|Language:|en|
|Size:|805.0 MB|

## References

https://huggingface.co/guymorganb/e5-large-v2-4096-lsg-patched