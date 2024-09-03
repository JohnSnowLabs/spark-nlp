---
layout: model
title: Burmese bge_base_financial_matryoshka_test_burmese_pipeline pipeline BGEEmbeddings from IlhamEbdesk
author: John Snow Labs
name: bge_base_financial_matryoshka_test_burmese_pipeline
date: 2024-09-01
tags: [my, open_source, pipeline, onnx]
task: Embeddings
language: my
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bge_base_financial_matryoshka_test_burmese_pipeline` is a Burmese model originally trained by IlhamEbdesk.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_base_financial_matryoshka_test_burmese_pipeline_my_5.4.2_3.0_1725198515678.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_base_financial_matryoshka_test_burmese_pipeline_my_5.4.2_3.0_1725198515678.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bge_base_financial_matryoshka_test_burmese_pipeline", lang = "my")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bge_base_financial_matryoshka_test_burmese_pipeline", lang = "my")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_base_financial_matryoshka_test_burmese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|my|
|Size:|375.9 MB|

## References

https://huggingface.co/IlhamEbdesk/bge-base-financial-matryoshka_test_my

## Included Models

- DocumentAssembler
- BGEEmbeddings