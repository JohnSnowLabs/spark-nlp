---
layout: model
title: English hubert_large_ll60k_librispeech_multi_gpu_pipeline pipeline HubertForCTC from r-sharma-coder
author: John Snow Labs
name: hubert_large_ll60k_librispeech_multi_gpu_pipeline
date: 2024-12-16
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_large_ll60k_librispeech_multi_gpu_pipeline` is a English model originally trained by r-sharma-coder.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_large_ll60k_librispeech_multi_gpu_pipeline_en_5.5.1_3.0_1734309111102.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_large_ll60k_librispeech_multi_gpu_pipeline_en_5.5.1_3.0_1734309111102.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hubert_large_ll60k_librispeech_multi_gpu_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hubert_large_ll60k_librispeech_multi_gpu_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hubert_large_ll60k_librispeech_multi_gpu_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|2.4 GB|

## References

https://huggingface.co/r-sharma-coder/hubert-large-ll60k-librispeech-multi-gpu

## Included Models

- AudioAssembler
- HubertForCTC