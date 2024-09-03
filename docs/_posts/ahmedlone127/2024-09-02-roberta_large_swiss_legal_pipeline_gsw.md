---
layout: model
title: Alemannic, Alsatian, Swiss German roberta_large_swiss_legal_pipeline pipeline RoBertaEmbeddings from joelito
author: John Snow Labs
name: roberta_large_swiss_legal_pipeline
date: 2024-09-02
tags: [gsw, open_source, pipeline, onnx]
task: Embeddings
language: gsw
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_large_swiss_legal_pipeline` is a Alemannic, Alsatian, Swiss German model originally trained by joelito.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_large_swiss_legal_pipeline_gsw_5.5.0_3.0_1725264317049.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_large_swiss_legal_pipeline_gsw_5.5.0_3.0_1725264317049.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_large_swiss_legal_pipeline", lang = "gsw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_large_swiss_legal_pipeline", lang = "gsw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_large_swiss_legal_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|gsw|
|Size:|1.6 GB|

## References

https://huggingface.co/joelito/legal-swiss-roberta-large

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings