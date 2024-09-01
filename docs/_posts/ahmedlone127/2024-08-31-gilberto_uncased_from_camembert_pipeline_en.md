---
layout: model
title: English gilberto_uncased_from_camembert_pipeline pipeline CamemBertEmbeddings from idb-ita
author: John Snow Labs
name: gilberto_uncased_from_camembert_pipeline
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

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gilberto_uncased_from_camembert_pipeline` is a English model originally trained by idb-ita.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gilberto_uncased_from_camembert_pipeline_en_5.4.2_3.0_1725130910114.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gilberto_uncased_from_camembert_pipeline_en_5.4.2_3.0_1725130910114.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gilberto_uncased_from_camembert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gilberto_uncased_from_camembert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gilberto_uncased_from_camembert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|262.9 MB|

## References

https://huggingface.co/idb-ita/gilberto-uncased-from-camembert

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings