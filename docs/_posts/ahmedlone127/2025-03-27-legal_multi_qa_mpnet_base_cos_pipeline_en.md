---
layout: model
title: English legal_multi_qa_mpnet_base_cos_pipeline pipeline MPNetEmbeddings from yuriyvnv
author: John Snow Labs
name: legal_multi_qa_mpnet_base_cos_pipeline
date: 2025-03-27
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`legal_multi_qa_mpnet_base_cos_pipeline` is a English model originally trained by yuriyvnv.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legal_multi_qa_mpnet_base_cos_pipeline_en_5.5.1_3.0_1743117533247.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legal_multi_qa_mpnet_base_cos_pipeline_en_5.5.1_3.0_1743117533247.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("legal_multi_qa_mpnet_base_cos_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("legal_multi_qa_mpnet_base_cos_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legal_multi_qa_mpnet_base_cos_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.3 MB|

## References

https://huggingface.co/yuriyvnv/legal-multi-qa-mpnet-base-cos

## Included Models

- DocumentAssembler
- MPNetEmbeddings