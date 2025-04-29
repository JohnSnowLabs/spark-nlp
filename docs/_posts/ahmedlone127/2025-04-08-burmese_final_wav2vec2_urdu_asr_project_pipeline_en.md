---
layout: model
title: English burmese_final_wav2vec2_urdu_asr_project_pipeline pipeline Wav2Vec2ForCTC from Raffay
author: John Snow Labs
name: burmese_final_wav2vec2_urdu_asr_project_pipeline
date: 2025-04-08
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`burmese_final_wav2vec2_urdu_asr_project_pipeline` is a English model originally trained by Raffay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/burmese_final_wav2vec2_urdu_asr_project_pipeline_en_5.5.1_3.0_1744129443147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/burmese_final_wav2vec2_urdu_asr_project_pipeline_en_5.5.1_3.0_1744129443147.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("burmese_final_wav2vec2_urdu_asr_project_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("burmese_final_wav2vec2_urdu_asr_project_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|burmese_final_wav2vec2_urdu_asr_project_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|348.7 MB|

## References

https://huggingface.co/Raffay/my_final_wav2vec2-urdu-asr-project

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC