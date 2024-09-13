---
layout: model
title: Finnish whisper_finnish_finetuned_small_200k_samples_pipeline pipeline WhisperForCTC from RASMUS
author: John Snow Labs
name: whisper_finnish_finetuned_small_200k_samples_pipeline
date: 2024-09-12
tags: [fi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_finnish_finetuned_small_200k_samples_pipeline` is a Finnish model originally trained by RASMUS.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_finnish_finetuned_small_200k_samples_pipeline_fi_5.5.0_3.0_1726139267349.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_finnish_finetuned_small_200k_samples_pipeline_fi_5.5.0_3.0_1726139267349.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_finnish_finetuned_small_200k_samples_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_finnish_finetuned_small_200k_samples_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_finnish_finetuned_small_200k_samples_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|1.7 GB|

## References

https://huggingface.co/RASMUS/Whisper_Finnish_finetuned_small_200k_samples

## Included Models

- AudioAssembler
- WhisperForCTC