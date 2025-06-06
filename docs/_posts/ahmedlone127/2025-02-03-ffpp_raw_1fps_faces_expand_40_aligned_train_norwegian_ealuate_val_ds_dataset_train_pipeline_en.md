---
layout: model
title: English ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline pipeline SwinForImageClassification from hchcsuim
author: John Snow Labs
name: ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline
date: 2025-02-03
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

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline` is a English model originally trained by hchcsuim.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline_en_5.5.1_3.0_1738570547922.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline_en_5.5.1_3.0_1738570547922.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ffpp_raw_1fps_faces_expand_40_aligned_train_norwegian_ealuate_val_ds_dataset_train_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|206.3 MB|

## References

https://huggingface.co/hchcsuim/FFPP-Raw_1FPS_faces-expand-40-aligned_train-no-ealuate_val-ds-dataset-train

## Included Models

- ImageAssembler
- SwinForImageClassification