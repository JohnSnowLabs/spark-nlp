---
layout: model
title: English deid_mdeberta_v3_base_finetuned_v3_pipeline pipeline DeBertaForTokenClassification from Sheripov
author: John Snow Labs
name: deid_mdeberta_v3_base_finetuned_v3_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DeBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deid_mdeberta_v3_base_finetuned_v3_pipeline` is a English model originally trained by Sheripov.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deid_mdeberta_v3_base_finetuned_v3_pipeline_en_5.5.1_3.0_1743134021514.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deid_mdeberta_v3_base_finetuned_v3_pipeline_en_5.5.1_3.0_1743134021514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deid_mdeberta_v3_base_finetuned_v3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deid_mdeberta_v3_base_finetuned_v3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deid_mdeberta_v3_base_finetuned_v3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|804.4 MB|

## References

https://huggingface.co/Sheripov/deid-mdeberta-v3-base-finetuned-v3

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForTokenClassification