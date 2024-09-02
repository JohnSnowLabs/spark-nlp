---
layout: model
title: English albert_base_v2_finetuned_squad_v2_hadsag_pipeline pipeline AlbertForQuestionAnswering from hadsag
author: John Snow Labs
name: albert_base_v2_finetuned_squad_v2_hadsag_pipeline
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

Pretrained AlbertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_base_v2_finetuned_squad_v2_hadsag_pipeline` is a English model originally trained by hadsag.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_v2_finetuned_squad_v2_hadsag_pipeline_en_5.5.0_3.0_1725237094533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_base_v2_finetuned_squad_v2_hadsag_pipeline_en_5.5.0_3.0_1725237094533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_base_v2_finetuned_squad_v2_hadsag_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_base_v2_finetuned_squad_v2_hadsag_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_v2_finetuned_squad_v2_hadsag_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|53.9 MB|

## References

https://huggingface.co/hadsag/albert-base-v2-finetuned-squad-v2

## Included Models

- MultiDocumentAssembler
- AlbertForQuestionAnswering