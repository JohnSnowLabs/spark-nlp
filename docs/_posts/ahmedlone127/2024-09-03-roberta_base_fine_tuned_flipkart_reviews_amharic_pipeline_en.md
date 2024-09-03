---
layout: model
title: English roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline pipeline XlmRoBertaForSequenceClassification from akshatmehta98
author: John Snow Labs
name: roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline
date: 2024-09-03
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline` is a English model originally trained by akshatmehta98.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline_en_5.5.0_3.0_1725327473396.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline_en_5.5.0_3.0_1725327473396.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_fine_tuned_flipkart_reviews_amharic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification