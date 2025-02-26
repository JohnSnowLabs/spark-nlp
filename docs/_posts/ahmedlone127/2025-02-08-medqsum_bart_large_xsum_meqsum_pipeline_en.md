---
layout: model
title: English medqsum_bart_large_xsum_meqsum_pipeline pipeline BartTransformer from NouRed
author: John Snow Labs
name: medqsum_bart_large_xsum_meqsum_pipeline
date: 2025-02-08
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`medqsum_bart_large_xsum_meqsum_pipeline` is a English model originally trained by NouRed.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/medqsum_bart_large_xsum_meqsum_pipeline_en_5.5.1_3.0_1739038824610.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/medqsum_bart_large_xsum_meqsum_pipeline_en_5.5.1_3.0_1739038824610.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("medqsum_bart_large_xsum_meqsum_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("medqsum_bart_large_xsum_meqsum_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|medqsum_bart_large_xsum_meqsum_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.9 GB|

## References

https://huggingface.co/NouRed/medqsum-bart-large-xsum-meqsum

## Included Models

- DocumentAssembler
- BartTransformer