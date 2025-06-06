---
layout: model
title: English robbert_2023_dutch_base_ft_nlp_xxl_pipeline pipeline RoBertaEmbeddings from helena-balabin
author: John Snow Labs
name: robbert_2023_dutch_base_ft_nlp_xxl_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robbert_2023_dutch_base_ft_nlp_xxl_pipeline` is a English model originally trained by helena-balabin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robbert_2023_dutch_base_ft_nlp_xxl_pipeline_en_5.5.1_3.0_1743127302385.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robbert_2023_dutch_base_ft_nlp_xxl_pipeline_en_5.5.1_3.0_1743127302385.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robbert_2023_dutch_base_ft_nlp_xxl_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robbert_2023_dutch_base_ft_nlp_xxl_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robbert_2023_dutch_base_ft_nlp_xxl_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|464.8 MB|

## References

https://huggingface.co/helena-balabin/robbert-2023-dutch-base-ft-nlp-xxl

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings