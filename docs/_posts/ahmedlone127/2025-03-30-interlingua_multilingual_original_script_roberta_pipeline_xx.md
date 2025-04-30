---
layout: model
title: Multilingual interlingua_multilingual_original_script_roberta_pipeline pipeline RoBertaEmbeddings from ibm
author: John Snow Labs
name: interlingua_multilingual_original_script_roberta_pipeline
date: 2025-03-30
tags: [xx, open_source, pipeline, onnx]
task: Embeddings
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`interlingua_multilingual_original_script_roberta_pipeline` is a Multilingual model originally trained by ibm.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/interlingua_multilingual_original_script_roberta_pipeline_xx_5.5.1_3.0_1743368924608.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/interlingua_multilingual_original_script_roberta_pipeline_xx_5.5.1_3.0_1743368924608.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("interlingua_multilingual_original_script_roberta_pipeline", lang = "xx")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("interlingua_multilingual_original_script_roberta_pipeline", lang = "xx")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|interlingua_multilingual_original_script_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|638.6 MB|

## References

References

https://huggingface.co/ibm/ia-multilingual-original-script-roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings