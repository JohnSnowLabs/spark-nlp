---
layout: model
title: English roberta_finetuned_subjqa_movies_2_holyofferings_pipeline pipeline RoBertaForQuestionAnswering from holyofferings
author: John Snow Labs
name: roberta_finetuned_subjqa_movies_2_holyofferings_pipeline
date: 2024-12-16
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_finetuned_subjqa_movies_2_holyofferings_pipeline` is a English model originally trained by holyofferings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_finetuned_subjqa_movies_2_holyofferings_pipeline_en_5.5.1_3.0_1734340693433.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_finetuned_subjqa_movies_2_holyofferings_pipeline_en_5.5.1_3.0_1734340693433.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_finetuned_subjqa_movies_2_holyofferings_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_finetuned_subjqa_movies_2_holyofferings_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_finetuned_subjqa_movies_2_holyofferings_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|464.0 MB|

## References

https://huggingface.co/holyofferings/roberta-finetuned-subjqa-movies_2

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering