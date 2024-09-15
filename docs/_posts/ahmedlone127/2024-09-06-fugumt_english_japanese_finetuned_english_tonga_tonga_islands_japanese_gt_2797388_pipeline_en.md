---
layout: model
title: English fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline pipeline MarianTransformer from VJ11
author: John Snow Labs
name: fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline
date: 2024-09-06
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline` is a English model originally trained by VJ11.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline_en_5.5.0_3.0_1725636474195.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline_en_5.5.0_3.0_1725636474195.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fugumt_english_japanese_finetuned_english_tonga_tonga_islands_japanese_gt_2797388_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|222.3 MB|

## References

https://huggingface.co/VJ11/fugumt-en-ja-finetuned-en-to-ja-gt-2797388

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer