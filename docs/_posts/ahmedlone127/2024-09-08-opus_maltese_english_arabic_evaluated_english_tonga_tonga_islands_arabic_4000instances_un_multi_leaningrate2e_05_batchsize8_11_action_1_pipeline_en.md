---
layout: model
title: English opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline pipeline MarianTransformer from meghazisofiane
author: John Snow Labs
name: opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline
date: 2024-09-08
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline` is a English model originally trained by meghazisofiane.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline_en_5.5.0_3.0_1725795604439.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline_en_5.5.0_3.0_1725795604439.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_maltese_english_arabic_evaluated_english_tonga_tonga_islands_arabic_4000instances_un_multi_leaningrate2e_05_batchsize8_11_action_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|528.7 MB|

## References

https://huggingface.co/meghazisofiane/opus-mt-en-ar-evaluated-en-to-ar-4000instances-un_multi-leaningRate2e-05-batchSize8-11-action-1

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer