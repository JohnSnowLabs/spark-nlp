---
layout: model
title: English opus_base_aon_tfidf_wce_unsampled_pipeline pipeline MarianTransformer from ethansimrm
author: John Snow Labs
name: opus_base_aon_tfidf_wce_unsampled_pipeline
date: 2024-09-11
tags: [en, open_source, pipeline, onnx]
task: Translation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opus_base_aon_tfidf_wce_unsampled_pipeline` is a English model originally trained by ethansimrm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_base_aon_tfidf_wce_unsampled_pipeline_en_5.5.0_3.0_1726038525815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_base_aon_tfidf_wce_unsampled_pipeline_en_5.5.0_3.0_1726038525815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opus_base_aon_tfidf_wce_unsampled_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opus_base_aon_tfidf_wce_unsampled_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_base_aon_tfidf_wce_unsampled_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|509.0 MB|

## References

https://huggingface.co/ethansimrm/opus_base_AoN_tfidf_wce_unsampled

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer