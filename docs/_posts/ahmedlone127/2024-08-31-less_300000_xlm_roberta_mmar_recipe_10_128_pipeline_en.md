---
layout: model
title: English less_300000_xlm_roberta_mmar_recipe_10_128_pipeline pipeline XlmRoBertaEmbeddings from CennetOguz
author: John Snow Labs
name: less_300000_xlm_roberta_mmar_recipe_10_128_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`less_300000_xlm_roberta_mmar_recipe_10_128_pipeline` is a English model originally trained by CennetOguz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/less_300000_xlm_roberta_mmar_recipe_10_128_pipeline_en_5.4.2_3.0_1725137972655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/less_300000_xlm_roberta_mmar_recipe_10_128_pipeline_en_5.4.2_3.0_1725137972655.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("less_300000_xlm_roberta_mmar_recipe_10_128_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("less_300000_xlm_roberta_mmar_recipe_10_128_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|less_300000_xlm_roberta_mmar_recipe_10_128_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/CennetOguz/less_300000_xlm_roberta_mmar_recipe_10_128

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings