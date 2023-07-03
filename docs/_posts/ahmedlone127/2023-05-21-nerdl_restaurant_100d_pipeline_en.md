---
layout: model
title: Pipeline to Detect Restaurant-related Terminology
author: John Snow Labs
name: nerdl_restaurant_100d_pipeline
date: 2023-05-21
tags: [restaurant, ner, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [nerdl_restaurant_100d](https://nlp.johnsnowlabs.com/2021/12/31/nerdl_restaurant_100d_en.html) model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_restaurant_100d_pipeline_en_4.4.2_3.0_1684650284287.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerdl_restaurant_100d_pipeline_en_4.4.2_3.0_1684650284287.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

restaurant_pipeline = PretrainedPipeline("nerdl_restaurant_100d_pipeline", lang = "en")

restaurant_pipeline.annotate("Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day.")
```
```scala

val restaurant_pipeline = new PretrainedPipeline("nerdl_restaurant_100d_pipeline", lang = "en")

restaurant_pipeline.annotate("Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day.")
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
restaurant_pipeline = PretrainedPipeline("nerdl_restaurant_100d_pipeline", lang = "en")

restaurant_pipeline.annotate("Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day.")
```
```scala
val restaurant_pipeline = new PretrainedPipeline("nerdl_restaurant_100d_pipeline", lang = "en")

restaurant_pipeline.annotate("Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day.")
```
</div>

## Results

```bash
Results



+---------------------------+---------------+
|chunk                      |ner_label      |
+---------------------------+---------------+
|Hong Kong’s                |Restaurant_Name|
|favourite                  |Rating         |
|pasta bar                  |Dish           |
|most reasonably            |Price          |
|lunch                      |Hours          |
|in town!                   |Location       |
|Sha Tin – Pici’s           |Restaurant_Name|
|burrata                    |Dish           |
|arugula salad              |Dish           |
|freshly tossed tuna tartare|Dish           |
|reliable                   |Price          |
|handmade pasta             |Dish           |
|pappardelle                |Dish           |
|effortless                 |Amenity        |
|Italian                    |Cuisine        |
|tidy one-pot               |Amenity        |
|espresso                   |Dish           |
+---------------------------+---------------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_restaurant_100d_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|166.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter
- Finisher