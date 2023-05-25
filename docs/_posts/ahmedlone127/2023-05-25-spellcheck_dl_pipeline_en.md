---
layout: model
title: Context Spell Checker Pipeline for English
author: John Snow Labs
name: spellcheck_dl_pipeline
date: 2023-05-25
tags: [spellcheck, spell, spellcheck_pipeline, spelling_corrector, en, open_source]
task: Spell Check
language: en
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained spellchecker pipeline is built on the top of [spellcheck_dl](https://nlp.johnsnowlabs.com/2022/04/02/spellcheck_dl_en_2_4.html) model. This pipeline is for PySpark 2.4.x users with SparkNLP 3.4.2 and above.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_dl_pipeline_en_4.4.2_3.4_1685008513632.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/spellcheck_dl_pipeline_en_4.4.2_3.4_1685008513632.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


pipeline = PretrainedPipeline("spellcheck_dl_pipeline", lang = "en")

text = ["During the summer we have the best ueather.", "I have a black ueather jacket, so nice."]

pipeline.annotate(text)
```
```scala


val pipeline = new PretrainedPipeline("spellcheck_dl_pipeline", lang = "en")

val example = Array("During the summer we have the best ueather.", "I have a black ueather jacket, so nice.")

pipeline.annotate(example)
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("spellcheck_dl_pipeline", lang = "en")

text = ["During the summer we have the best ueather.", "I have a black ueather jacket, so nice."]

pipeline.annotate(text)
```
```scala
val pipeline = new PretrainedPipeline("spellcheck_dl_pipeline", lang = "en")

val example = Array("During the summer we have the best ueather.", "I have a black ueather jacket, so nice.")

pipeline.annotate(example)
```
</div>

## Results

```bash
Results




[{'checked': ['During', 'the', 'summer', 'we', 'have', 'the', 'best', 'weather', '.'],
  'document': ['During the summer we have the best ueather.'],
  'token': ['During', 'the', 'summer', 'we', 'have', 'the', 'best', 'ueather', '.']},

 {'checked': ['I', 'have', 'a', 'black', 'leather', 'jacket', ',', 'so', 'nice',  '.'],
  'document': ['I have a black ueather jacket, so nice.'],
  'token': ['I', 'have', 'a', 'black', 'ueather', 'jacket', ',', 'so', 'nice', '.']}]


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_dl_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|99.7 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ContextSpellCheckerModel