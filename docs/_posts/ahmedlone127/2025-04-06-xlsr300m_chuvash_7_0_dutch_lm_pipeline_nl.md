---
layout: model
title: Dutch, Flemish xlsr300m_chuvash_7_0_dutch_lm_pipeline pipeline Wav2Vec2ForCTC from Iskaj
author: John Snow Labs
name: xlsr300m_chuvash_7_0_dutch_lm_pipeline
date: 2025-04-06
tags: [nl, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlsr300m_chuvash_7_0_dutch_lm_pipeline` is a Dutch, Flemish model originally trained by Iskaj.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlsr300m_chuvash_7_0_dutch_lm_pipeline_nl_5.5.1_3.0_1743935298789.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlsr300m_chuvash_7_0_dutch_lm_pipeline_nl_5.5.1_3.0_1743935298789.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlsr300m_chuvash_7_0_dutch_lm_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlsr300m_chuvash_7_0_dutch_lm_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlsr300m_chuvash_7_0_dutch_lm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|1.2 GB|

## References

https://huggingface.co/Iskaj/xlsr300m_cv_7.0_nl_lm

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC