---
layout: model
title: English ai_club_inductions_21_nlp_distilbert_pipeline pipeline DistilBertForQuestionAnswering from AyushPJ
author: John Snow Labs
name: ai_club_inductions_21_nlp_distilbert_pipeline
date: 2024-09-08
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ai_club_inductions_21_nlp_distilbert_pipeline` is a English model originally trained by AyushPJ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ai_club_inductions_21_nlp_distilbert_pipeline_en_5.5.0_3.0_1725798560299.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ai_club_inductions_21_nlp_distilbert_pipeline_en_5.5.0_3.0_1725798560299.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ai_club_inductions_21_nlp_distilbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ai_club_inductions_21_nlp_distilbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ai_club_inductions_21_nlp_distilbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/AyushPJ/ai-club-inductions-21-nlp-distilBERT

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering