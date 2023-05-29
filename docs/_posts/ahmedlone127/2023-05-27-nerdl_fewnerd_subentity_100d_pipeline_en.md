---
layout: model
title: Pre-trained Pipeline for Few-NERD NER Model
author: John Snow Labs
name: nerdl_fewnerd_subentity_100d_pipeline
date: 2023-05-27
tags: [fewnerd, ner, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.4.2
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on Few-NERD/inter public dataset and it extracts 66 entities that are in general scope.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_subentity_100d_pipeline_en_4.4.2_3.2_1685204361691.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_subentity_100d_pipeline_en_4.4.2_3.2_1685204361691.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

fewnerd_pipeline = PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

fewnerd_pipeline.annotate("""12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.""")
```
```scala

val pipeline = new PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

val result = pipeline.fullAnnotate("12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.")(0)
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
fewnerd_pipeline = PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

fewnerd_pipeline.annotate("""12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.""")
```
```scala
val pipeline = new PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

val result = pipeline.fullAnnotate("12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.")(0)
```
</div>

## Results

```bash
Results



+-----------------------+----------------------------+
|chunk                  |ner_label                   |
+-----------------------+----------------------------+
|Corazones ('12 Hearts')|art-broadcastprogram        |
|Spanish-language       |other-language              |
|United States          |location-GPE                |
|Telemundo              |organization-media/newspaper|
|Argentine TV           |organization-media/newspaper|
|Los Angeles            |location-GPE                |
|Steven Spielberg       |person-director             |
|Cloverfield Paradox    |art-film                    |
+-----------------------+----------------------------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_fewnerd_subentity_100d_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|167.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter
- Finisher