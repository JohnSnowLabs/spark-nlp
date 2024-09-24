---
layout: model
title: English distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline pipeline RoBertaEmbeddings from ethanoutangoun
author: John Snow Labs
name: distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline
date: 2024-09-21
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline` is a English model originally trained by ethanoutangoun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline_en_5.5.0_3.0_1726934785651.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline_en_5.5.0_3.0_1726934785651.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilroberta_base_finetuned_wikitext2_ethanoutangoun_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|306.5 MB|

## References

https://huggingface.co/ethanoutangoun/distilroberta-base-finetuned-wikitext2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings