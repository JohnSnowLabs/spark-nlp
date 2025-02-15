---
layout: model
title: Hindi mbert_hindi_telugu_mlm_pipeline pipeline BertEmbeddings from hapandya
author: John Snow Labs
name: mbert_hindi_telugu_mlm_pipeline
date: 2025-02-08
tags: [hi, open_source, pipeline, onnx]
task: Embeddings
language: hi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_hindi_telugu_mlm_pipeline` is a Hindi model originally trained by hapandya.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_hindi_telugu_mlm_pipeline_hi_5.5.1_3.0_1739019678948.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_hindi_telugu_mlm_pipeline_hi_5.5.1_3.0_1739019678948.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbert_hindi_telugu_mlm_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbert_hindi_telugu_mlm_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbert_hindi_telugu_mlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|663.2 MB|

## References

https://huggingface.co/hapandya/mBERT-hi-te-MLM

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings