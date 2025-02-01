---
layout: model
title: English vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline pipeline ViTForImageClassification from ansilmbabl
author: John Snow Labs
name: vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline
date: 2025-02-01
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline` is a English model originally trained by ansilmbabl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline_en_5.5.1_3.0_1738377606574.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline_en_5.5.1_3.0_1738377606574.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vit_base_patch16_224_in21k_cards_june_08_cropping_filtered_preprocess_change_test_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/ansilmbabl/vit-base-patch16-224-in21k-cards-june-08-cropping-filtered-preprocess-change-test-2

## Included Models

- ImageAssembler
- ViTForImageClassification