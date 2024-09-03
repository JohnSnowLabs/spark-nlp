---
layout: model
title: English clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline pipeline CLIPForZeroShotClassification from Aixile
author: John Snow Labs
name: clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline
date: 2024-09-02
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
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

Pretrained CLIPForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline` is a English model originally trained by Aixile.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline_en_5.5.0_3.0_1725276263511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline_en_5.5.0_3.0_1725276263511.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clip_vit_l_14_datacomp_xl_s13b_b90k_aixile_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/Aixile/CLIP-ViT-L-14-DataComp.XL-s13B-b90K

## Included Models

- ImageAssembler
- CLIPForZeroShotClassification