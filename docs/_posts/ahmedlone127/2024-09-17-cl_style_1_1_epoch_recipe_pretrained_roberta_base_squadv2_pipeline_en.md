---
layout: model
title: English cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline pipeline RoBertaForQuestionAnswering from AnonymousSub
author: John Snow Labs
name: cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline
date: 2024-09-17
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline` is a English model originally trained by AnonymousSub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline_en_5.5.0_3.0_1726580653649.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline_en_5.5.0_3.0_1726580653649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cl_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.3 MB|

## References

https://huggingface.co/AnonymousSub/CL_style_1_1_epoch_recipe_pretrained_roberta_base_squadv2

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering