---
layout: model
title: Bulgarian roberta_base_bulgarian_pipeline pipeline RoBertaEmbeddings from iarfmoose
author: John Snow Labs
name: roberta_base_bulgarian_pipeline
date: 2024-09-01
tags: [bg, open_source, pipeline, onnx]
task: Embeddings
language: bg
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_bulgarian_pipeline` is a Bulgarian model originally trained by iarfmoose.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_bulgarian_pipeline_bg_5.4.2_3.0_1725192044184.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_bulgarian_pipeline_bg_5.4.2_3.0_1725192044184.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_bulgarian_pipeline", lang = "bg")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_bulgarian_pipeline", lang = "bg")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_bulgarian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|bg|
|Size:|471.1 MB|

## References

https://huggingface.co/iarfmoose/roberta-base-bulgarian

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings