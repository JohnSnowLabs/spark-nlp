---
layout: model
title: English phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline pipeline MPNetEmbeddings from dbourget
author: John Snow Labs
name: phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline
date: 2024-09-10
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline` is a English model originally trained by dbourget.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline_en_5.5.0_3.0_1725963848860.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline_en_5.5.0_3.0_1725963848860.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|phil_sim_sentence_transformers_all_mpnet_base_v2_2024_03_11_21_44_34_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/dbourget/phil-sim-sentence-transformers-all-mpnet-base-v2-2024-03-11_21-44-34

## Included Models

- DocumentAssembler
- MPNetEmbeddings