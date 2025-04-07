---
layout: model
title: Polish xtreme_s_xlsr_mls_upd_pipeline pipeline Wav2Vec2ForCTC from anton-l
author: John Snow Labs
name: xtreme_s_xlsr_mls_upd_pipeline
date: 2025-04-05
tags: [pl, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: pl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xtreme_s_xlsr_mls_upd_pipeline` is a Polish model originally trained by anton-l.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xtreme_s_xlsr_mls_upd_pipeline_pl_5.5.1_3.0_1743872331888.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xtreme_s_xlsr_mls_upd_pipeline_pl_5.5.1_3.0_1743872331888.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xtreme_s_xlsr_mls_upd_pipeline", lang = "pl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xtreme_s_xlsr_mls_upd_pipeline", lang = "pl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xtreme_s_xlsr_mls_upd_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pl|
|Size:|1.2 GB|

## References

https://huggingface.co/anton-l/xtreme_s_xlsr_mls_upd

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC