---
layout: model
title: English base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline pipeline WhisperForCTC from saahith
author: John Snow Labs
name: base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline` is a English model originally trained by saahith.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline_en_5.5.0_3.0_1726950055300.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline_en_5.5.0_3.0_1726950055300.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|base_english_final_combined_2_0_8_1e_05_balmy_sweep_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|646.6 MB|

## References

https://huggingface.co/saahith/base.en-final-combined-2-0-8-1e-05-balmy-sweep-1

## Included Models

- AudioAssembler
- WhisperForCTC