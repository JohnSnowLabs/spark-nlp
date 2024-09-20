---
layout: model
title: English usclm_distilbert_base_uncased_mk1_pipeline pipeline DistilBertEmbeddings from hyperdemocracy
author: John Snow Labs
name: usclm_distilbert_base_uncased_mk1_pipeline
date: 2024-09-08
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`usclm_distilbert_base_uncased_mk1_pipeline` is a English model originally trained by hyperdemocracy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/usclm_distilbert_base_uncased_mk1_pipeline_en_5.5.0_3.0_1725775999667.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/usclm_distilbert_base_uncased_mk1_pipeline_en_5.5.0_3.0_1725775999667.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("usclm_distilbert_base_uncased_mk1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("usclm_distilbert_base_uncased_mk1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|usclm_distilbert_base_uncased_mk1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/hyperdemocracy/usclm-distilbert-base-uncased-mk1

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings