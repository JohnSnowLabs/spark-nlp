---
layout: model
title: Letzeburgesch, Luxembourgish luxembourgishstt_pipeline pipeline Wav2Vec2ForCTC from Parsavares
author: John Snow Labs
name: luxembourgishstt_pipeline
date: 2025-04-06
tags: [lb, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: lb
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`luxembourgishstt_pipeline` is a Letzeburgesch, Luxembourgish model originally trained by Parsavares.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/luxembourgishstt_pipeline_lb_5.5.1_3.0_1743959989949.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/luxembourgishstt_pipeline_lb_5.5.1_3.0_1743959989949.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("luxembourgishstt_pipeline", lang = "lb")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("luxembourgishstt_pipeline", lang = "lb")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|luxembourgishstt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|lb|
|Size:|233.1 MB|

## References

https://huggingface.co/Parsavares/LuxembourgishSTT

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC