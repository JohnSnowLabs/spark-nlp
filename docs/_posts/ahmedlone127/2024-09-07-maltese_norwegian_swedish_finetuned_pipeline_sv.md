---
layout: model
title: Swedish maltese_norwegian_swedish_finetuned_pipeline pipeline MarianTransformer from oskarandrsson
author: John Snow Labs
name: maltese_norwegian_swedish_finetuned_pipeline
date: 2024-09-07
tags: [sv, open_source, pipeline, onnx]
task: Translation
language: sv
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`maltese_norwegian_swedish_finetuned_pipeline` is a Swedish model originally trained by oskarandrsson.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/maltese_norwegian_swedish_finetuned_pipeline_sv_5.5.0_3.0_1725747442822.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/maltese_norwegian_swedish_finetuned_pipeline_sv_5.5.0_3.0_1725747442822.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("maltese_norwegian_swedish_finetuned_pipeline", lang = "sv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("maltese_norwegian_swedish_finetuned_pipeline", lang = "sv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|maltese_norwegian_swedish_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sv|
|Size:|204.8 MB|

## References

https://huggingface.co/oskarandrsson/mt-no-sv-finetuned

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer