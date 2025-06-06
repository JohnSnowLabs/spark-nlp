---
layout: model
title: English e5_base_unsupervised_pseudo_gpl_fiqa_131a12 E5Embeddings from rithwik-db
author: John Snow Labs
name: e5_base_unsupervised_pseudo_gpl_fiqa_131a12
date: 2024-12-17
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

Pretrained E5Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`e5_base_unsupervised_pseudo_gpl_fiqa_131a12` is a English model originally trained by rithwik-db.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_base_unsupervised_pseudo_gpl_fiqa_131a12_en_5.5.1_3.0_1734396845056.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_base_unsupervised_pseudo_gpl_fiqa_131a12_en_5.5.1_3.0_1734396845056.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
embeddings = E5Embeddings.pretrained("e5_base_unsupervised_pseudo_gpl_fiqa_131a12","en") \
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
    
val embeddings = E5Embeddings.pretrained("e5_base_unsupervised_pseudo_gpl_fiqa_131a12","en") 
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
|Model Name:|e5_base_unsupervised_pseudo_gpl_fiqa_131a12|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[E5]|
|Language:|en|
|Size:|402.7 MB|

## References

https://huggingface.co/rithwik-db/e5-base-unsupervised-pseudo-gpl-fiqa-131a12