---
layout: model
title: Swedish nordic_roberta_wiki_pipeline pipeline RoBertaEmbeddings from flax-community
author: John Snow Labs
name: nordic_roberta_wiki_pipeline
date: 2024-09-10
tags: [sv, open_source, pipeline, onnx]
task: Embeddings
language: sv
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nordic_roberta_wiki_pipeline` is a Swedish model originally trained by flax-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nordic_roberta_wiki_pipeline_sv_5.5.0_3.0_1726005604514.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nordic_roberta_wiki_pipeline_sv_5.5.0_3.0_1726005604514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nordic_roberta_wiki_pipeline", lang = "sv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nordic_roberta_wiki_pipeline", lang = "sv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nordic_roberta_wiki_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sv|
|Size:|465.6 MB|

## References

https://huggingface.co/flax-community/nordic-roberta-wiki

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings