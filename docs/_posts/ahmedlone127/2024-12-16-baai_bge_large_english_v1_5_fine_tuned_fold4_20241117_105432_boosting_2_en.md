---
layout: model
title: English baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2 BGEEmbeddings from marumarukun
author: John Snow Labs
name: baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2
date: 2024-12-16
tags: [en, open_source, onnx, embeddings, bge]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BGEEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2` is a English model originally trained by marumarukun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2_en_5.5.1_3.0_1734359033680.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2_en_5.5.1_3.0_1734359033680.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

embeddings = BGEEmbeddings.pretrained("baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2","en") \
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
    

val embeddings = BGEEmbeddings.pretrained("baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))
val data = Seq("I love spark-nlp).toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|baai_bge_large_english_v1_5_fine_tuned_fold4_20241117_105432_boosting_2|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[bge]|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/marumarukun/BAAI-bge-large-en-v1.5_fine_tuned_fold4_20241117_105432_boosting_2