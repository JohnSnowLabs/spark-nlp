---
layout: model
title: Portuguese mdeberta_v3_base_assin2_similarity_pipeline pipeline DeBertaForSequenceClassification from ruanchaves
author: John Snow Labs
name: mdeberta_v3_base_assin2_similarity_pipeline
date: 2024-09-01
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_v3_base_assin2_similarity_pipeline` is a Portuguese model originally trained by ruanchaves.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_assin2_similarity_pipeline_pt_5.5.0_3.0_1725210284001.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_assin2_similarity_pipeline_pt_5.5.0_3.0_1725210284001.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdeberta_v3_base_assin2_similarity_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdeberta_v3_base_assin2_similarity_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_v3_base_assin2_similarity_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|836.2 MB|

## References

https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-similarity

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification