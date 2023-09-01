---
layout: model
title: NER Pipeline for 10 African Languages
author: John Snow Labs
name: xlm_roberta_large_token_classifier_masakhaner_pipeline
date: 2023-05-25
tags: [masakhaner, african, xlm_roberta, multilingual, pipeline, amharic, hausa, igbo, kinyarwanda, luganda, swahilu, wolof, yoruba, nigerian, pidgin, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on [xlm_roberta_large_token_classifier_masakhaner](https://nlp.johnsnowlabs.com/2021/12/06/xlm_roberta_large_token_classifier_masakhaner_xx.html) ner model which is imported from `HuggingFace`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_masakhaner_pipeline_xx_4.4.2_3.4_1685006535738.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_masakhaner_pipeline_xx_4.4.2_3.4_1685006535738.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
masakhaner_pipeline = PretrainedPipeline("xlm_roberta_large_token_classifier_masakhaner_pipeline", lang = "xx")

masakhaner_pipeline.annotate("አህመድ ቫንዳ ከ3-10-2000 ጀምሮ በአዲስ አበባ ኖሯል።")
```
```scala
val masakhaner_pipeline = new PretrainedPipeline("xlm_roberta_large_token_classifier_masakhaner_pipeline", lang = "xx")

val masakhaner_pipeline.annotate("አህመድ ቫንዳ ከ3-10-2000 ጀምሮ በአዲስ አበባ ኖሯል።")
```
</div>

## Results

```bash
Results



+----------------+---------+
|chunk           |ner_label|
+----------------+---------+
|አህመድ ቫንዳ      |PER      |
|ከ3-10-2000 ጀምሮ|DATE      |
|በአዲስ አበባ       |LOC      |
+----------------+---------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_masakhaner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.8 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- XlmRoBertaForTokenClassification
- NerConverter
- Finisher