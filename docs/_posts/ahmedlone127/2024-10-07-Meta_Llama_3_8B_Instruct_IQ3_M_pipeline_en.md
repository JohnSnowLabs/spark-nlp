---
layout: model
title: English Meta_Llama_3_8B_Instruct_IQ3_M_pipeline pipeline AutoGGUFModel from lmstudio-community
author: John Snow Labs
name: Meta_Llama_3_8B_Instruct_IQ3_M_pipeline
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

Pretrained AutoGGUFModel, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`Meta_Llama_3_8B_Instruct_IQ3_M_pipeline` is a English model originally trained by lmstudio-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Meta_Llama_3_8B_Instruct_IQ3_M_pipeline_en_5.5.1_3.0_1728343063962.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Meta_Llama_3_8B_Instruct_IQ3_M_pipeline_en_5.5.1_3.0_1728343063962.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("Meta_Llama_3_8B_Instruct_IQ3_M_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("Meta_Llama_3_8B_Instruct_IQ3_M_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Meta_Llama_3_8B_Instruct_IQ3_M_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|3.8 GB|

## References

https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF

## Included Models

- DocumentAssembler
- AutoGGUFModel