---
layout: model
title: English gemma_2_2b_it_iq3_m_pipeline pipeline AutoGGUFModel from lmstudio-community
author: John Snow Labs
name: gemma_2_2b_it_iq3_m_pipeline
date: 2024-10-07
tags: [en, open_source, pipeline, onnx]
task: Text Generation
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

Pretrained AutoGGUFModel, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gemma_2_2b_it_iq3_m_pipeline` is a English model originally trained by lmstudio-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gemma_2_2b_it_iq3_m_pipeline_en_5.5.1_3.0_1728345158207.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gemma_2_2b_it_iq3_m_pipeline_en_5.5.1_3.0_1728345158207.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gemma_2_2b_it_iq3_m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gemma_2_2b_it_iq3_m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gemma_2_2b_it_iq3_m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.4 GB|

## References

https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF

## Included Models

- DocumentAssembler
- AutoGGUFModel