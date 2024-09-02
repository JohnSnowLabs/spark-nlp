---
layout: model
title: Swahili (macrolanguage) roberta_base_wechsel_swahili_pipeline pipeline RoBertaEmbeddings from benjamin
author: John Snow Labs
name: roberta_base_wechsel_swahili_pipeline
date: 2024-09-01
tags: [sw, open_source, pipeline, onnx]
task: Embeddings
language: sw
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_wechsel_swahili_pipeline` is a Swahili (macrolanguage) model originally trained by benjamin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_swahili_pipeline_sw_5.4.2_3.0_1725191244854.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_swahili_pipeline_sw_5.4.2_3.0_1725191244854.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_wechsel_swahili_pipeline", lang = "sw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_wechsel_swahili_pipeline", lang = "sw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_wechsel_swahili_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|sw|
|Size:|465.7 MB|

## References

https://huggingface.co/benjamin/roberta-base-wechsel-swahili

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings