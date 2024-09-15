---
layout: model
title: English tnana_english_thai_align_finetuned_pipeline pipeline MarianTransformer from SAI2-EXP
author: John Snow Labs
name: tnana_english_thai_align_finetuned_pipeline
date: 2024-09-07
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tnana_english_thai_align_finetuned_pipeline` is a English model originally trained by SAI2-EXP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tnana_english_thai_align_finetuned_pipeline_en_5.5.0_3.0_1725740736184.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tnana_english_thai_align_finetuned_pipeline_en_5.5.0_3.0_1725740736184.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tnana_english_thai_align_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tnana_english_thai_align_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tnana_english_thai_align_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|530.8 MB|

## References

https://huggingface.co/SAI2-EXP/TNANA-en-th-align-finetuned

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer