---
layout: model
title: English aiornot_profile_picture_fine_tuned_pipeline pipeline SwinForImageClassification from TLeonidas
author: John Snow Labs
name: aiornot_profile_picture_fine_tuned_pipeline
date: 2025-02-05
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

Pretrained SwinForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`aiornot_profile_picture_fine_tuned_pipeline` is a English model originally trained by TLeonidas.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/aiornot_profile_picture_fine_tuned_pipeline_en_5.5.1_3.0_1738751740470.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/aiornot_profile_picture_fine_tuned_pipeline_en_5.5.1_3.0_1738751740470.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("aiornot_profile_picture_fine_tuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("aiornot_profile_picture_fine_tuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|aiornot_profile_picture_fine_tuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|206.3 MB|

## References

https://huggingface.co/TLeonidas/AIorNot-Profile-Picture-Fine-Tuned

## Included Models

- ImageAssembler
- SwinForImageClassification