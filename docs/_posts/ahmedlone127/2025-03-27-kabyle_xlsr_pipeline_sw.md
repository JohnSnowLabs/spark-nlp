---
layout: model
title: Swahili (macrolanguage) kabyle_xlsr_pipeline pipeline Wav2Vec2ForCTC from Akashpb13
author: John Snow Labs
name: kabyle_xlsr_pipeline
date: 2025-03-27
tags: [sw, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: sw
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kabyle_xlsr_pipeline` is a Swahili (macrolanguage) model originally trained by Akashpb13.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kabyle_xlsr_pipeline_sw_5.5.1_3.0_1743097710472.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kabyle_xlsr_pipeline_sw_5.5.1_3.0_1743097710472.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kabyle_xlsr_pipeline", lang = "sw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kabyle_xlsr_pipeline", lang = "sw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kabyle_xlsr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|sw|
|Size:|1.2 GB|

## References

https://huggingface.co/Akashpb13/Kabyle_xlsr

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC