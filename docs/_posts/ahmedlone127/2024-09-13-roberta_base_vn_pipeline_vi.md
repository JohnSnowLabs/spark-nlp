---
layout: model
title: Vietnamese roberta_base_vn_pipeline pipeline RoBertaEmbeddings from NlpHUST
author: John Snow Labs
name: roberta_base_vn_pipeline
date: 2024-09-13
tags: [vi, open_source, pipeline, onnx]
task: Embeddings
language: vi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_vn_pipeline` is a Vietnamese model originally trained by NlpHUST.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_vn_pipeline_vi_5.5.0_3.0_1726264861061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_vn_pipeline_vi_5.5.0_3.0_1726264861061.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_vn_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_vn_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_vn_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|466.2 MB|

## References

https://huggingface.co/NlpHUST/roberta-base-vn

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings