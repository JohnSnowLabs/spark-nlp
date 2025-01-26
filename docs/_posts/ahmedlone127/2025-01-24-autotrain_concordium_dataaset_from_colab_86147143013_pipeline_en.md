---
layout: model
title: English autotrain_concordium_dataaset_from_colab_86147143013_pipeline pipeline BertForQuestionAnswering from hasle1975
author: John Snow Labs
name: autotrain_concordium_dataaset_from_colab_86147143013_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_concordium_dataaset_from_colab_86147143013_pipeline` is a English model originally trained by hasle1975.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_concordium_dataaset_from_colab_86147143013_pipeline_en_5.5.1_3.0_1737691155094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_concordium_dataaset_from_colab_86147143013_pipeline_en_5.5.1_3.0_1737691155094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_concordium_dataaset_from_colab_86147143013_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_concordium_dataaset_from_colab_86147143013_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_concordium_dataaset_from_colab_86147143013_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/hasle1975/autotrain-concordium-dataaset-from-colab-86147143013

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering