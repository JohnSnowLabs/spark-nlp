---
layout: model
title: English pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline pipeline GPT2Transformer from jvelja
author: John Snow Labs
name: pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline
date: 2025-01-30
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline` is a English model originally trained by jvelja.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline_en_5.5.1_3.0_1738262842077.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline_en_5.5.1_3.0_1738262842077.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pythia_finetune_gpt2_clusters_4_nobsgc_lr_0_0005_modularity_ravel_mixedfixcluster_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|469.7 MB|

## References

https://huggingface.co/jvelja/pythia-finetune-gpt2-clusters-4-NoBSGC-lr_0.0005-Modularity-RAVEL_MIXEDFixCluster

## Included Models

- DocumentAssembler
- GPT2Transformer