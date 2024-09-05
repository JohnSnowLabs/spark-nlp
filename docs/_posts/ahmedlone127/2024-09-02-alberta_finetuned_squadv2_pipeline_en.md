---
layout: model
title: English alberta_finetuned_squadv2_pipeline pipeline AlbertForQuestionAnswering from quocviethere
author: John Snow Labs
name: alberta_finetuned_squadv2_pipeline
date: 2024-09-02
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

Pretrained AlbertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`alberta_finetuned_squadv2_pipeline` is a English model originally trained by quocviethere.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/alberta_finetuned_squadv2_pipeline_en_5.5.0_3.0_1725310018165.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/alberta_finetuned_squadv2_pipeline_en_5.5.0_3.0_1725310018165.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("alberta_finetuned_squadv2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("alberta_finetuned_squadv2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|alberta_finetuned_squadv2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|42.0 MB|

## References

https://huggingface.co/quocviethere/alberta-finetuned-squadv2

## Included Models

- MultiDocumentAssembler
- AlbertForQuestionAnswering