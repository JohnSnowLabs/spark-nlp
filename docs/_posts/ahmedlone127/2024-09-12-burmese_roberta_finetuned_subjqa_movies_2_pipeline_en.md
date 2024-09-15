---
layout: model
title: English burmese_roberta_finetuned_subjqa_movies_2_pipeline pipeline RoBertaForQuestionAnswering from vmg1957
author: John Snow Labs
name: burmese_roberta_finetuned_subjqa_movies_2_pipeline
date: 2024-09-12
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`burmese_roberta_finetuned_subjqa_movies_2_pipeline` is a English model originally trained by vmg1957.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/burmese_roberta_finetuned_subjqa_movies_2_pipeline_en_5.5.0_3.0_1726183908834.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/burmese_roberta_finetuned_subjqa_movies_2_pipeline_en_5.5.0_3.0_1726183908834.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("burmese_roberta_finetuned_subjqa_movies_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("burmese_roberta_finetuned_subjqa_movies_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|burmese_roberta_finetuned_subjqa_movies_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|464.1 MB|

## References

https://huggingface.co/vmg1957/my-roberta-finetuned-subjqa-movies_2

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering