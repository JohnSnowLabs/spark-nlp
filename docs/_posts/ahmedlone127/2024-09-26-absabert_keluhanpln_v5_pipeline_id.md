---
layout: model
title: Indonesian absabert_keluhanpln_v5_pipeline pipeline BertForSequenceClassification from radityapranata
author: John Snow Labs
name: absabert_keluhanpln_v5_pipeline
date: 2024-09-26
tags: [id, open_source, pipeline, onnx]
task: Text Classification
language: id
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`absabert_keluhanpln_v5_pipeline` is a Indonesian model originally trained by radityapranata.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/absabert_keluhanpln_v5_pipeline_id_5.5.0_3.0_1727361826815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/absabert_keluhanpln_v5_pipeline_id_5.5.0_3.0_1727361826815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("absabert_keluhanpln_v5_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("absabert_keluhanpln_v5_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|absabert_keluhanpln_v5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|409.4 MB|

## References

https://huggingface.co/radityapranata/absabert-keluhanpln-v5

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification