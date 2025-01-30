---
layout: model
title: Indonesian indobert_large_p2_finetuned_sayula_popoluca_pipeline pipeline BertForTokenClassification from ageng-anugrah
author: John Snow Labs
name: indobert_large_p2_finetuned_sayula_popoluca_pipeline
date: 2025-01-26
tags: [id, open_source, pipeline, onnx]
task: Named Entity Recognition
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indobert_large_p2_finetuned_sayula_popoluca_pipeline` is a Indonesian model originally trained by ageng-anugrah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indobert_large_p2_finetuned_sayula_popoluca_pipeline_id_5.5.1_3.0_1737934303921.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indobert_large_p2_finetuned_sayula_popoluca_pipeline_id_5.5.1_3.0_1737934303921.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indobert_large_p2_finetuned_sayula_popoluca_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indobert_large_p2_finetuned_sayula_popoluca_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indobert_large_p2_finetuned_sayula_popoluca_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|1.3 GB|

## References

https://huggingface.co/ageng-anugrah/indobert-large-p2-finetuned-pos

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification