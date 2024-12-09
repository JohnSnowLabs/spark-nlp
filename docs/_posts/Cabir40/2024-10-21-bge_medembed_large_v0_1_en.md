---
layout: model
title: English bge_medembed_large_v0_1 BGEEmbeddings from abhinand
author: John Snow Labs
name: bge_medembed_large_v0_1
date: 2024-10-21
tags: [embedding, en, open_source, bge, medical, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BGEEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. 
`bge_medembed_large_v0_1` is a English model originally trained by abhinand

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_medembed_large_v0_1_en_5.5.0_3.0_1729515260623.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_medembed_large_v0_1_en_5.5.0_3.0_1729515260623.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

embeddings = BGEEmbeddings.pretrained("bge_medembed_large_v0_1","en")\
      .setInputCols(["document"])\
      .setOutputCol("embeddings")       
        
pipeline = Pipeline(
    stages = [
        document_assembler, 
        embeddings
])

data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val embeddings = BGEEmbeddings.pretrained("bge_medembed_large_v0_1","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val data = Seq("I love spark-nlp").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash

+----------------------------------------------------------------------------------------------------+
|                                                                                       bge_embedding|
+----------------------------------------------------------------------------------------------------+
|[{sentence_embeddings, 0, 15, I love spark-nlp, {sentence -> 0}, [-0.018065551, -0.032784615, 0.0...|
+----------------------------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_medembed_large_v0_1|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[bge]|
|Language:|en|
|Size:|1.2 GB|