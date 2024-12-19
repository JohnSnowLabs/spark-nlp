---
layout: model
title: English hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline pipeline HubertForCTC from macabdul9
author: John Snow Labs
name: hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline
date: 2024-12-19
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

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline` is a English model originally trained by macabdul9.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline_en_5.5.1_3.0_1734600165333.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline_en_5.5.1_3.0_1734600165333.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hubert_large_ll60k_librispeech_clean_100h_demo_dist_macabdul9_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|2.4 GB|

## References

https://huggingface.co/macabdul9/hubert-large-ll60k-librispeech-clean-100h-demo-dist

## Included Models

- AudioAssembler
- HubertForCTC