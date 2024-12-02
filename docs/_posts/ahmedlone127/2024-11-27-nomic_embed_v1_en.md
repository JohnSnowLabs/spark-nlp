---
layout: model
title: nomic_embed_v1 model from nomic-ai
author: John Snow Labs
name: nomic_embed_v1
date: 2024-11-27
tags: [en, open_source, openvino]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: NomicEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained NomicEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mini_cpm_2b_8bit` is a multilingual model originally trained by openbmb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nomic_embed_v1_en_5.5.1_3.0_1732743647389.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nomic_embed_v1_en_5.5.1_3.0_1732743647389.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
embeddings = NomicEmbeddings.pretrained("nomic_embed_v1","en") \
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
    
val embeddings = NomicEmbeddings.pretrained("nomic_embed_v1","en") 
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
|Model Name:|nomic_embed_v1|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|255.0 MB|

## References

https://huggingface.co/nomic-ai/nomic-embed-text-v1