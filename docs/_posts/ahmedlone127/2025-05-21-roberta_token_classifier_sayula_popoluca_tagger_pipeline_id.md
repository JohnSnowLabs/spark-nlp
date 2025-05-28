---
layout: model
title: Indonesian roberta_token_classifier_sayula_popoluca_tagger_pipeline pipeline RoBertaForTokenClassification from w11wo
author: John Snow Labs
name: roberta_token_classifier_sayula_popoluca_tagger_pipeline
date: 2025-05-21
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_token_classifier_sayula_popoluca_tagger_pipeline` is a Indonesian model originally trained by w11wo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_sayula_popoluca_tagger_pipeline_id_5.5.1_3.0_1747857837421.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_sayula_popoluca_tagger_pipeline_id_5.5.1_3.0_1747857837421.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_token_classifier_sayula_popoluca_tagger_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_token_classifier_sayula_popoluca_tagger_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_sayula_popoluca_tagger_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|465.6 MB|

## References

https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification