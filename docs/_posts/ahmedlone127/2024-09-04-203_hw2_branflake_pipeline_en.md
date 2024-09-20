---
layout: model
title: English 203_hw2_branflake_pipeline pipeline RoBertaForQuestionAnswering from branflake
author: John Snow Labs
name: 203_hw2_branflake_pipeline
date: 2024-09-04
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`203_hw2_branflake_pipeline` is a English model originally trained by branflake.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/203_hw2_branflake_pipeline_en_5.5.0_3.0_1725483891814.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/203_hw2_branflake_pipeline_en_5.5.0_3.0_1725483891814.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("203_hw2_branflake_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("203_hw2_branflake_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|203_hw2_branflake_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|464.0 MB|

## References

https://huggingface.co/branflake/203_hw2

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering