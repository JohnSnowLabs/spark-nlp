---
layout: model
title: English bio_roberta_base_pubmed_pipeline pipeline RoBertaEmbeddings from minhpqn
author: John Snow Labs
name: bio_roberta_base_pubmed_pipeline
date: 2024-09-02
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bio_roberta_base_pubmed_pipeline` is a English model originally trained by minhpqn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bio_roberta_base_pubmed_pipeline_en_5.5.0_3.0_1725265128854.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bio_roberta_base_pubmed_pipeline_en_5.5.0_3.0_1725265128854.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bio_roberta_base_pubmed_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bio_roberta_base_pubmed_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bio_roberta_base_pubmed_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.0 MB|

## References

https://huggingface.co/minhpqn/bio_roberta-base_pubmed

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings