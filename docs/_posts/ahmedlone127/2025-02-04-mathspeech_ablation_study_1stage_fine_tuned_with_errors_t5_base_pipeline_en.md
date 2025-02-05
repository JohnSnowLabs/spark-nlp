---
layout: model
title: English mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline pipeline T5Transformer from AAAI2025
author: John Snow Labs
name: mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline
date: 2025-02-04
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline` is a English model originally trained by AAAI2025.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline_en_5.5.1_3.0_1738702141016.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline_en_5.5.1_3.0_1738702141016.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mathspeech_ablation_study_1stage_fine_tuned_with_errors_t5_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|927.6 MB|

## References

https://huggingface.co/AAAI2025/MathSpeech_Ablation_Study_1stage_fine-tuned_with_errors_T5_base

## Included Models

- DocumentAssembler
- T5Transformer