---
layout: model
title: English samsumgen_xsum_conv_cl_coda_seed102_pipeline pipeline BartTransformer from PSW
author: John Snow Labs
name: samsumgen_xsum_conv_cl_coda_seed102_pipeline
date: 2025-04-07
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`samsumgen_xsum_conv_cl_coda_seed102_pipeline` is a English model originally trained by PSW.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/samsumgen_xsum_conv_cl_coda_seed102_pipeline_en_5.5.1_3.0_1744014219181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/samsumgen_xsum_conv_cl_coda_seed102_pipeline_en_5.5.1_3.0_1744014219181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("samsumgen_xsum_conv_cl_coda_seed102_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("samsumgen_xsum_conv_cl_coda_seed102_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|samsumgen_xsum_conv_cl_coda_seed102_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|810.3 MB|

## References

https://huggingface.co/PSW/samsumgen-xsum-conv-cl-coda-seed102

## Included Models

- DocumentAssembler
- BartTransformer