---
layout: model
title: English ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline pipeline Wav2Vec2ForCTC from gary109
author: John Snow Labs
name: ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline
date: 2025-04-07
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline` is a English model originally trained by gary109.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline_en_5.5.1_3.0_1744010391329.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline_en_5.5.1_3.0_1744010391329.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ai_light_dance_drums_ft_pretrain_wav2vec2_base_nepal_bhasa_onset_idmt_mdb_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|353.6 MB|

## References

https://huggingface.co/gary109/ai-light-dance_drums_ft_pretrain_wav2vec2-base-new_onset-idmt-mdb-2

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC