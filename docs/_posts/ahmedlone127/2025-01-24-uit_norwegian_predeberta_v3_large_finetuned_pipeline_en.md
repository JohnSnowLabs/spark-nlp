---
layout: model
title: English uit_norwegian_predeberta_v3_large_finetuned_pipeline pipeline DeBertaForSequenceClassification from sercetexam9
author: John Snow Labs
name: uit_norwegian_predeberta_v3_large_finetuned_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`uit_norwegian_predeberta_v3_large_finetuned_pipeline` is a English model originally trained by sercetexam9.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/uit_norwegian_predeberta_v3_large_finetuned_pipeline_en_5.5.1_3.0_1737728412998.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/uit_norwegian_predeberta_v3_large_finetuned_pipeline_en_5.5.1_3.0_1737728412998.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("uit_norwegian_predeberta_v3_large_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("uit_norwegian_predeberta_v3_large_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|uit_norwegian_predeberta_v3_large_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/sercetexam9/UIT-NO-PREdeberta-v3-large-finetuned

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification