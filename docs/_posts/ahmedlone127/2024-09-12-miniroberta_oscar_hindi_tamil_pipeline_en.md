---
layout: model
title: English miniroberta_oscar_hindi_tamil_pipeline pipeline RoBertaEmbeddings from neibyr
author: John Snow Labs
name: miniroberta_oscar_hindi_tamil_pipeline
date: 2024-09-12
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`miniroberta_oscar_hindi_tamil_pipeline` is a English model originally trained by neibyr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/miniroberta_oscar_hindi_tamil_pipeline_en_5.5.0_3.0_1726112944663.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/miniroberta_oscar_hindi_tamil_pipeline_en_5.5.0_3.0_1726112944663.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("miniroberta_oscar_hindi_tamil_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("miniroberta_oscar_hindi_tamil_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|miniroberta_oscar_hindi_tamil_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|347.1 MB|

## References

https://huggingface.co/neibyr/MiniRoberta_oscar_hindi_tamil

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings