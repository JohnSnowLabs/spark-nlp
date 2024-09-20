---
layout: model
title: Portuguese mminilm_l6_v2_mmarco_v2_pipeline pipeline XlmRoBertaForSequenceClassification from unicamp-dl
author: John Snow Labs
name: mminilm_l6_v2_mmarco_v2_pipeline
date: 2024-09-03
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mminilm_l6_v2_mmarco_v2_pipeline` is a Portuguese model originally trained by unicamp-dl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mminilm_l6_v2_mmarco_v2_pipeline_pt_5.5.0_3.0_1725396666381.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mminilm_l6_v2_mmarco_v2_pipeline_pt_5.5.0_3.0_1725396666381.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mminilm_l6_v2_mmarco_v2_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mminilm_l6_v2_mmarco_v2_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mminilm_l6_v2_mmarco_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|368.5 MB|

## References

https://huggingface.co/unicamp-dl/mMiniLM-L6-v2-mmarco-v2

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification