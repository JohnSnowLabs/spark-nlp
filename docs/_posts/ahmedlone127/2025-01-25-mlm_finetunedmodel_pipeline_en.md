---
layout: model
title: English mlm_finetunedmodel_pipeline pipeline RoBertaEmbeddings from shradha01
author: John Snow Labs
name: mlm_finetunedmodel_pipeline
date: 2025-01-25
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mlm_finetunedmodel_pipeline` is a English model originally trained by shradha01.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mlm_finetunedmodel_pipeline_en_5.5.1_3.0_1737797003245.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mlm_finetunedmodel_pipeline_en_5.5.1_3.0_1737797003245.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mlm_finetunedmodel_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mlm_finetunedmodel_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mlm_finetunedmodel_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.1 MB|

## References

https://huggingface.co/shradha01/MLM_FinetunedModel

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings