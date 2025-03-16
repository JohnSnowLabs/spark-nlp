---
layout: model
title: English mfds_all_mpnet_base_v2 MPNetEmbeddings from pedutronix
author: John Snow Labs
name: mfds_all_mpnet_base_v2
date: 2025-01-23
tags: [en, open_source, onnx, embeddings, mpnet]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mfds_all_mpnet_base_v2` is a English model originally trained by pedutronix.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mfds_all_mpnet_base_v2_en_5.5.1_3.0_1737669551632.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mfds_all_mpnet_base_v2_en_5.5.1_3.0_1737669551632.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
embeddings = MPNetEmbeddings.pretrained("mfds_all_mpnet_base_v2","en") \
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
    
val embeddings = MPNetEmbeddings.pretrained("mfds_all_mpnet_base_v2","en") 
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
|Model Name:|mfds_all_mpnet_base_v2|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[mpnet]|
|Language:|en|
|Size:|406.7 MB|

## References

https://huggingface.co/pedutronix/mfds-all-mpnet-base-v2