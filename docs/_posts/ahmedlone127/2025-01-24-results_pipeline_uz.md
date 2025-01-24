---
layout: model
title: Uzbek results_pipeline pipeline XlmRoBertaForTokenClassification from Xojakbar
author: John Snow Labs
name: results_pipeline
date: 2025-01-24
tags: [uz, open_source, pipeline, onnx]
task: Named Entity Recognition
language: uz
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`results_pipeline` is a Uzbek model originally trained by Xojakbar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/results_pipeline_uz_5.5.1_3.0_1737679850612.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/results_pipeline_uz_5.5.1_3.0_1737679850612.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("results_pipeline", lang = "uz")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("results_pipeline", lang = "uz")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|results_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uz|
|Size:|819.0 MB|

## References

https://huggingface.co/Xojakbar/results

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification