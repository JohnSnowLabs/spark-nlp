---
layout: model
title: English finetunned_legal_deberta_v3_base_pipeline pipeline DeBertaForQuestionAnswering from satishsingh90
author: John Snow Labs
name: finetunned_legal_deberta_v3_base_pipeline
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

Pretrained DeBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetunned_legal_deberta_v3_base_pipeline` is a English model originally trained by satishsingh90.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetunned_legal_deberta_v3_base_pipeline_en_5.5.0_3.0_1725240299329.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetunned_legal_deberta_v3_base_pipeline_en_5.5.0_3.0_1725240299329.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetunned_legal_deberta_v3_base_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetunned_legal_deberta_v3_base_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetunned_legal_deberta_v3_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|569.2 MB|

## References

https://huggingface.co/satishsingh90/finetunned_legal_deberta_v3_base

## Included Models

- MultiDocumentAssembler
- DeBertaForQuestionAnswering