---
layout: model
title: English asr_alffamharic_phoneme_based_pipeline pipeline Wav2Vec2ForCTC from Samuael
author: John Snow Labs
name: asr_alffamharic_phoneme_based_pipeline
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_alffamharic_phoneme_based_pipeline` is a English model originally trained by Samuael.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_alffamharic_phoneme_based_pipeline_en_5.5.1_3.0_1744050910000.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_alffamharic_phoneme_based_pipeline_en_5.5.1_3.0_1744050910000.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("asr_alffamharic_phoneme_based_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("asr_alffamharic_phoneme_based_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_alffamharic_phoneme_based_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|348.7 MB|

## References

https://huggingface.co/Samuael/asr-alffamharic-phoneme-based

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC