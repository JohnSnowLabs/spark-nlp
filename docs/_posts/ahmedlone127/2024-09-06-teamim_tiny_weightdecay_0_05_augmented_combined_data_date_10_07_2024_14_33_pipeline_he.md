---
layout: model
title: Hebrew teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline pipeline WhisperForCTC from cantillation
author: John Snow Labs
name: teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline
date: 2024-09-06
tags: [he, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: he
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline` is a Hebrew model originally trained by cantillation.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline_he_5.5.0_3.0_1725582757562.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline_he_5.5.0_3.0_1725582757562.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|teamim_tiny_weightdecay_0_05_augmented_combined_data_date_10_07_2024_14_33_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|388.9 MB|

## References

https://huggingface.co/cantillation/Teamim-tiny_WeightDecay-0.05_Augmented_Combined-Data_date-10-07-2024_14-33

## Included Models

- AudioAssembler
- WhisperForCTC