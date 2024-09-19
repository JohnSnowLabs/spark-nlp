---
layout: model
title: English twitter_roberta_base_finetuned_twitter_user_desc_pipeline pipeline RoBertaEmbeddings from bdotloh
author: John Snow Labs
name: twitter_roberta_base_finetuned_twitter_user_desc_pipeline
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`twitter_roberta_base_finetuned_twitter_user_desc_pipeline` is a English model originally trained by bdotloh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/twitter_roberta_base_finetuned_twitter_user_desc_pipeline_en_5.5.0_3.0_1726006457238.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/twitter_roberta_base_finetuned_twitter_user_desc_pipeline_en_5.5.0_3.0_1726006457238.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("twitter_roberta_base_finetuned_twitter_user_desc_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("twitter_roberta_base_finetuned_twitter_user_desc_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_roberta_base_finetuned_twitter_user_desc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.0 MB|

## References

https://huggingface.co/bdotloh/twitter-roberta-base-finetuned-twitter-user-desc

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings