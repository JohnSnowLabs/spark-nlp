---
layout: model
title: English kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline pipeline MarianTransformer from srvmishra832
author: John Snow Labs
name: kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline
date: 2025-04-06
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline` is a English model originally trained by srvmishra832.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline_en_5.5.1_3.0_1743971636480.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline_en_5.5.1_3.0_1743971636480.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kde4_dataset_translation_english_tonga_tonga_islands_hindi_with_opus_maltese_english_hindi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.5 MB|

## References

https://huggingface.co/srvmishra832/KDE4_Dataset_Translation_English_to_Hindi_with_opus_mt_en_hi

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer