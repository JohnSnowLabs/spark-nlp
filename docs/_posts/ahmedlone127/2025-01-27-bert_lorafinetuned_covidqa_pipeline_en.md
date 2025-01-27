---
layout: model
title: English bert_lorafinetuned_covidqa_pipeline pipeline BertEmbeddings from dqduong2003
author: John Snow Labs
name: bert_lorafinetuned_covidqa_pipeline
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_lorafinetuned_covidqa_pipeline` is a English model originally trained by dqduong2003.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_lorafinetuned_covidqa_pipeline_en_5.5.1_3.0_1737954118383.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_lorafinetuned_covidqa_pipeline_en_5.5.1_3.0_1737954118383.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_lorafinetuned_covidqa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_lorafinetuned_covidqa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_lorafinetuned_covidqa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

https://huggingface.co/dqduong2003/bert-lorafinetuned-covidqa

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings