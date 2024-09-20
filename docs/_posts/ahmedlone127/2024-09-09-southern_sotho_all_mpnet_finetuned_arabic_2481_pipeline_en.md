---
layout: model
title: English southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline pipeline MPNetEmbeddings from danfeg
author: John Snow Labs
name: southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline` is a English model originally trained by danfeg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline_en_5.5.0_3.0_1725875142239.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline_en_5.5.0_3.0_1725875142239.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|southern_sotho_all_mpnet_finetuned_arabic_2481_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.9 MB|

## References

https://huggingface.co/danfeg/ST-ALL-MPNET_Finetuned-AR-2481

## Included Models

- DocumentAssembler
- MPNetEmbeddings