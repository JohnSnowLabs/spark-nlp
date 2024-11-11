---
layout: model
title: Thai whisper_thai_small_combined_finetuned_pipeline pipeline WhisperForCTC from pother
author: John Snow Labs
name: whisper_thai_small_combined_finetuned_pipeline
date: 2024-11-11
tags: [th, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: th
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_thai_small_combined_finetuned_pipeline` is a Thai model originally trained by pother.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_thai_small_combined_finetuned_pipeline_th_5.5.1_3.0_1731343063475.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_thai_small_combined_finetuned_pipeline_th_5.5.1_3.0_1731343063475.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_thai_small_combined_finetuned_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_thai_small_combined_finetuned_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_thai_small_combined_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|1.7 GB|

## References

https://huggingface.co/pother/whisper-th-small-combined-finetuned

## Included Models

- AudioAssembler
- WhisperForCTC