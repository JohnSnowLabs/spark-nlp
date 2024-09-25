---
layout: model
title: English code_bert_small_finetuned_v2_pipeline pipeline RoBertaEmbeddings from mshn74
author: John Snow Labs
name: code_bert_small_finetuned_v2_pipeline
date: 2024-09-19
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`code_bert_small_finetuned_v2_pipeline` is a English model originally trained by mshn74.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/code_bert_small_finetuned_v2_pipeline_en_5.5.0_3.0_1726747098257.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/code_bert_small_finetuned_v2_pipeline_en_5.5.0_3.0_1726747098257.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("code_bert_small_finetuned_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("code_bert_small_finetuned_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|code_bert_small_finetuned_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|311.8 MB|

## References

https://huggingface.co/mshn74/code_bert_small-finetuned-v2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings