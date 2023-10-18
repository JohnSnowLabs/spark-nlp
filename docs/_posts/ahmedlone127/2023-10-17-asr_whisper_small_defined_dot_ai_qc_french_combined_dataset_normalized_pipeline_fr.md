---
layout: model
title: French asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline pipeline WhisperForCTC from jwkritchie
author: John Snow Labs
name: asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline
date: 2023-10-17
tags: [whisper, fr, open_source, pipeline]
task: Automatic Speech Recognition
language: fr
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline` is a French model originally trained by jwkritchie.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline_fr_5.1.4_3.4_1697582437910.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline_fr_5.1.4_3.4_1697582437910.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline', lang = 'fr')
annotations =  pipeline.transform(audioDF)

```
```scala

val pipeline = new PretrainedPipeline('asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline', lang = 'fr')
val annotations = pipeline.transform(audioDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_small_defined_dot_ai_qc_french_combined_dataset_normalized_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|1.7 GB|

## References

https://huggingface.co/jwkritchie/whisper-small-defined-dot-ai-qc-fr-combined-dataset-normalized

## Included Models

- AudioAssembler
- WhisperForCTC