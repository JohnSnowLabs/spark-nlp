---
layout: model
title: Catalan, Valencian catt5_solr_finetunned_pipeline pipeline T5Transformer from oooriii
author: John Snow Labs
name: catt5_solr_finetunned_pipeline
date: 2024-08-26
tags: [ca, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ca
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`catt5_solr_finetunned_pipeline` is a Catalan, Valencian model originally trained by oooriii.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/catt5_solr_finetunned_pipeline_ca_5.4.2_3.0_1724666780264.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/catt5_solr_finetunned_pipeline_ca_5.4.2_3.0_1724666780264.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("catt5_solr_finetunned_pipeline", lang = "ca")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("catt5_solr_finetunned_pipeline", lang = "ca")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|catt5_solr_finetunned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ca|
|Size:|917.8 MB|

## References

https://huggingface.co/oooriii/catt5-solr-finetunned

## Included Models

- DocumentAssembler
- T5Transformer