---
layout: model
title: Ukrainian ukr_roberta_base_pipeline pipeline RoBertaEmbeddings from youscan
author: John Snow Labs
name: ukr_roberta_base_pipeline
date: 2024-09-02
tags: [uk, open_source, pipeline, onnx]
task: Embeddings
language: uk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ukr_roberta_base_pipeline` is a Ukrainian model originally trained by youscan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ukr_roberta_base_pipeline_uk_5.5.0_3.0_1725264760220.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ukr_roberta_base_pipeline_uk_5.5.0_3.0_1725264760220.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ukr_roberta_base_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ukr_roberta_base_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ukr_roberta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|471.3 MB|

## References

https://huggingface.co/youscan/ukr-roberta-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings