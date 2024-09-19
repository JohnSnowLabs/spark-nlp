---
layout: model
title: English movie_review_roberta_imalexianne_pipeline pipeline RoBertaForSequenceClassification from imalexianne
author: John Snow Labs
name: movie_review_roberta_imalexianne_pipeline
date: 2024-09-07
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`movie_review_roberta_imalexianne_pipeline` is a English model originally trained by imalexianne.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/movie_review_roberta_imalexianne_pipeline_en_5.5.0_3.0_1725717691094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/movie_review_roberta_imalexianne_pipeline_en_5.5.0_3.0_1725717691094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("movie_review_roberta_imalexianne_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("movie_review_roberta_imalexianne_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|movie_review_roberta_imalexianne_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|463.6 MB|

## References

https://huggingface.co/imalexianne/Movie_Review_Roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification