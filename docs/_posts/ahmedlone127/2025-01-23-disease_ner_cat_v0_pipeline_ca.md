---
layout: model
title: Catalan, Valencian disease_ner_cat_v0_pipeline pipeline RoBertaForTokenClassification from BSC-NLP4BIA
author: John Snow Labs
name: disease_ner_cat_v0_pipeline
date: 2025-01-23
tags: [ca, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ca
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`disease_ner_cat_v0_pipeline` is a Catalan, Valencian model originally trained by BSC-NLP4BIA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/disease_ner_cat_v0_pipeline_ca_5.5.1_3.0_1737666532337.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/disease_ner_cat_v0_pipeline_ca_5.5.1_3.0_1737666532337.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("disease_ner_cat_v0_pipeline", lang = "ca")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("disease_ner_cat_v0_pipeline", lang = "ca")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|disease_ner_cat_v0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ca|
|Size:|436.1 MB|

## References

https://huggingface.co/BSC-NLP4BIA/disease-ner-cat-v0

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification