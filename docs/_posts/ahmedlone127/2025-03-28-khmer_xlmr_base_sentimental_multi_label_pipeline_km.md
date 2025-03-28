---
layout: model
title: Central Khmer, Khmer khmer_xlmr_base_sentimental_multi_label_pipeline pipeline XlmRoBertaForSequenceClassification from songhieng
author: John Snow Labs
name: khmer_xlmr_base_sentimental_multi_label_pipeline
date: 2025-03-28
tags: [km, open_source, pipeline, onnx]
task: Text Classification
language: km
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`khmer_xlmr_base_sentimental_multi_label_pipeline` is a Central Khmer, Khmer model originally trained by songhieng.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/khmer_xlmr_base_sentimental_multi_label_pipeline_km_5.5.1_3.0_1743173366371.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/khmer_xlmr_base_sentimental_multi_label_pipeline_km_5.5.1_3.0_1743173366371.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("khmer_xlmr_base_sentimental_multi_label_pipeline", lang = "km")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("khmer_xlmr_base_sentimental_multi_label_pipeline", lang = "km")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|khmer_xlmr_base_sentimental_multi_label_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|km|
|Size:|817.1 MB|

## References

https://huggingface.co/songhieng/khmer-xlmr-base-sentimental-multi-label

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification