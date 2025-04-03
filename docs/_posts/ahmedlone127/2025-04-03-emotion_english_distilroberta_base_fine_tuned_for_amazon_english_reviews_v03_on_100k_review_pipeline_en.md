---
layout: model
title: English emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline pipeline RoBertaForSequenceClassification from Abdelrahman-Rezk
author: John Snow Labs
name: emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline
date: 2025-04-03
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline` is a English model originally trained by Abdelrahman-Rezk.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline_en_5.5.1_3.0_1743693917832.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline_en_5.5.1_3.0_1743693917832.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|emotion_english_distilroberta_base_fine_tuned_for_amazon_english_reviews_v03_on_100k_review_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|309.1 MB|

## References

https://huggingface.co/Abdelrahman-Rezk/emotion-english-distilroberta-base-fine_tuned_for_amazon_english_reviews_V03_on_100K_review

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification