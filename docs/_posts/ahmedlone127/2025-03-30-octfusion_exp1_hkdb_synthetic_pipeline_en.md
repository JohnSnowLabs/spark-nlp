---
layout: model
title: English octfusion_exp1_hkdb_synthetic_pipeline pipeline ViTForImageClassification from g30rv17ys
author: John Snow Labs
name: octfusion_exp1_hkdb_synthetic_pipeline
date: 2025-03-30
tags: [en, open_source, pipeline, onnx]
task: Image Classification
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`octfusion_exp1_hkdb_synthetic_pipeline` is a English model originally trained by g30rv17ys.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/octfusion_exp1_hkdb_synthetic_pipeline_en_5.5.1_3.0_1743373616295.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/octfusion_exp1_hkdb_synthetic_pipeline_en_5.5.1_3.0_1743373616295.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("octfusion_exp1_hkdb_synthetic_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("octfusion_exp1_hkdb_synthetic_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|octfusion_exp1_hkdb_synthetic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/g30rv17ys/OCTFusion-Exp1-HKDB-Synthetic

## Included Models

- ImageAssembler
- ViTForImageClassification