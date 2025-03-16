---
layout: model
title: English xlm_v_base_trimmed_spanish_pipeline pipeline XlmRoBertaEmbeddings from vocabtrimmer
author: John Snow Labs
name: xlm_v_base_trimmed_spanish_pipeline
date: 2025-01-27
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

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_v_base_trimmed_spanish_pipeline` is a English model originally trained by vocabtrimmer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_v_base_trimmed_spanish_pipeline_en_5.5.1_3.0_1737977344990.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_v_base_trimmed_spanish_pipeline_en_5.5.1_3.0_1737977344990.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_v_base_trimmed_spanish_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_v_base_trimmed_spanish_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_v_base_trimmed_spanish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|624.5 MB|

## References

https://huggingface.co/vocabtrimmer/xlm-v-base-trimmed-es

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings