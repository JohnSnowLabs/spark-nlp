---
layout: model
title: English xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline pipeline CLIPForZeroShotClassification from StanfordAIMI
author: John Snow Labs
name: xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline
date: 2025-01-23
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
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

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline` is a English model originally trained by StanfordAIMI.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline_en_5.5.1_3.0_1737631150135.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline_en_5.5.1_3.0_1737631150135.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xrayclip__vit_l_14__laion2b_s32b_b82k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/StanfordAIMI/XrayCLIP__vit-l-14__laion2b-s32b-b82k

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification