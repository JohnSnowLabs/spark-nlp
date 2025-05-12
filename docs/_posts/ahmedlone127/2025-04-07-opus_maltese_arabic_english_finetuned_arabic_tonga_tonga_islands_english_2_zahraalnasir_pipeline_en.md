---
layout: model
title: English opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline pipeline MarianTransformer from ZahraALnasir
author: John Snow Labs
name: opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline
date: 2025-04-07
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline` is a English model originally trained by ZahraALnasir.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline_en_5.5.1_3.0_1744018710742.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline_en_5.5.1_3.0_1744018710742.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_maltese_arabic_english_finetuned_arabic_tonga_tonga_islands_english_2_zahraalnasir_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|410.3 MB|

## References

https://huggingface.co/ZahraALnasir/opus-mt-ar-en-finetuned-ar-to-en-2

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer