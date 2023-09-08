---
layout: model
title: French biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr MPNetEmbeddings from antoinelouis
author: John Snow Labs
name: biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr
date: 2023-09-07
tags: [mpnet, fr, open_source, onnx]
task: Embeddings
language: fr
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

Pretrained MPNetEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr` is a French model originally trained by antoinelouis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr_fr_5.1.1_3.0_1694130498418.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr_fr_5.1.1_3.0_1694130498418.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =MPNetEmbeddings.pretrained("biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr","fr") \
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
    .pretrained("biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr", "fr")
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
|Model Name:|biencoder_multi_qa_mpnet_base_cos_v1_mmarcofr|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[mpnet_embeddings]|
|Language:|fr|
|Size:|407.0 MB|

## References

https://huggingface.co/antoinelouis/biencoder-multi-qa-mpnet-base-cos-v1-mmarcoFR