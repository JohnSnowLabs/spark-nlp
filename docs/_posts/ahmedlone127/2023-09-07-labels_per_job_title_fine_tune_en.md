---
layout: model
title: English labels_per_job_title_fine_tune MPNetEmbeddings from marianodo
author: John Snow Labs
name: labels_per_job_title_fine_tune
date: 2023-09-07
tags: [mpnet, en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.1
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`labels_per_job_title_fine_tune` is a English model originally trained by marianodo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/labels_per_job_title_fine_tune_en_5.1.1_3.0_1694125014401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/labels_per_job_title_fine_tune_en_5.1.1_3.0_1694125014401.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =MPNetEmbeddings.pretrained("labels_per_job_title_fine_tune","en") \
            .setInputCols(["documents"]) \
            .setOutputCol("mpnet_embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("documents")
    
val embeddings = MPNetEmbeddings 
    .pretrained("labels_per_job_title_fine_tune", "en")
    .setInputCols(Array("documents")) 
    .setOutputCol("mpnet_embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|labels_per_job_title_fine_tune|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[mpnet_embeddings]|
|Language:|en|
|Size:|406.8 MB|

## References

https://huggingface.co/marianodo/labels-per-job-title-fine-tune