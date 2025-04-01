---
layout: model
title: English nnn_bnft_64_0029_fnec_pipeline pipeline CamemBertForTokenClassification from StrangeSX
author: John Snow Labs
name: nnn_bnft_64_0029_fnec_pipeline
date: 2025-04-01
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

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nnn_bnft_64_0029_fnec_pipeline` is a English model originally trained by StrangeSX.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nnn_bnft_64_0029_fnec_pipeline_en_5.5.1_3.0_1743469039776.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nnn_bnft_64_0029_fnec_pipeline_en_5.5.1_3.0_1743469039776.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nnn_bnft_64_0029_fnec_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nnn_bnft_64_0029_fnec_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nnn_bnft_64_0029_fnec_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|392.2 MB|

## References

https://huggingface.co/StrangeSX/NNN-BNFT-64-0029-fnec

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification