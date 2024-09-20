---
layout: model
title: Modern Greek (1453-) sent_greeksocialbert_base_greek_uncased_v1_pipeline pipeline BertSentenceEmbeddings from gealexandri
author: John Snow Labs
name: sent_greeksocialbert_base_greek_uncased_v1_pipeline
date: 2024-09-18
tags: [el, open_source, pipeline, onnx]
task: Embeddings
language: el
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_greeksocialbert_base_greek_uncased_v1_pipeline` is a Modern Greek (1453-) model originally trained by gealexandri.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_greeksocialbert_base_greek_uncased_v1_pipeline_el_5.5.0_3.0_1726694224140.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_greeksocialbert_base_greek_uncased_v1_pipeline_el_5.5.0_3.0_1726694224140.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_greeksocialbert_base_greek_uncased_v1_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_greeksocialbert_base_greek_uncased_v1_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_greeksocialbert_base_greek_uncased_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|421.8 MB|

## References

https://huggingface.co/gealexandri/greeksocialbert-base-greek-uncased-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings