---
layout: model
title: English roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline pipeline RoBertaEmbeddings from msu-ceco
author: John Snow Labs
name: roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline
date: 2024-09-13
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline` is a English model originally trained by msu-ceco.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline_en_5.5.0_3.0_1726264490094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline_en_5.5.0_3.0_1726264490094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ft_on_agextcorpus_2023_12_10_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|465.9 MB|

## References

https://huggingface.co/msu-ceco/roberta-ft-on-agextcorpus-2023-12-10_v2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings