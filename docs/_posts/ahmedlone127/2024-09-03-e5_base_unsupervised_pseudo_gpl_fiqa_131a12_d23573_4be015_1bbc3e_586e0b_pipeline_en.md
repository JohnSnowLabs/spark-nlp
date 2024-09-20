---
layout: model
title: English e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline pipeline E5Embeddings from rithwik-db
author: John Snow Labs
name: e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline
date: 2024-09-03
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

Pretrained E5Embeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline` is a English model originally trained by rithwik-db.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline_en_5.5.0_3.0_1725393219140.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline_en_5.5.0_3.0_1725393219140.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|e5_base_unsupervised_pseudo_gpl_fiqa_131a12_d23573_4be015_1bbc3e_586e0b_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|402.6 MB|

## References

https://huggingface.co/rithwik-db/e5-base-unsupervised-pseudo-gpl-fiqa-131a12-d23573-4be015-1bbc3e-586e0b

## Included Models

- DocumentAssembler
- E5Embeddings