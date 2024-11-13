---
layout: model
title: Ukrainian hubert_ukrainian_pipeline pipeline HubertForCTC from Yehor
author: John Snow Labs
name: hubert_ukrainian_pipeline
date: 2024-11-08
tags: [uk, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: uk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_ukrainian_pipeline` is a Ukrainian model originally trained by Yehor.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_ukrainian_pipeline_uk_5.5.1_3.0_1731106461400.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_ukrainian_pipeline_uk_5.5.1_3.0_1731106461400.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hubert_ukrainian_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hubert_ukrainian_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hubert_ukrainian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|708.6 MB|

## References

https://huggingface.co/Yehor/hubert-uk

## Included Models

- AudioAssembler
- HubertForCTC