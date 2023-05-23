---
layout: model
title: Pretrained Pipeline for Few-NERD-General NER Model
author: John Snow Labs
name: nerdl_fewnerd_100d_pipeline
date: 2023-05-22
tags: [fewnerd, nerdl, en, open_source]
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

This pretrained pipeline is built on Few-NERD model and it detects :

`PERSON`, `ORGANIZATION`, `LOCATION`, `ART`, `BUILDING`, `PRODUCT`, `EVENT`, `OTHER`

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_100d_pipeline_en_4.4.2_3.2_1684759421066.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_100d_pipeline_en_4.4.2_3.2_1684759421066.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

fewnerd_pipeline = PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

fewnerd_pipeline.annotate("""The Double Down is a sandwich offered by Kentucky Fried Chicken restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).""")
```
```scala

val pipeline = new PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

val result = pipeline.fullAnnotate("The Double Down is a sandwich offered by Kentucky Fried Chicken restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).")(0)
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
fewnerd_pipeline = PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

fewnerd_pipeline.annotate("""The Double Down is a sandwich offered by Kentucky Fried Chicken restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).""")
```
```scala
val pipeline = new PretrainedPipeline("nerdl_fewnerd_subentity_100d_pipeline", lang = "en")

val result = pipeline.fullAnnotate("The Double Down is a sandwich offered by Kentucky Fried Chicken restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).")(0)
```
</div>

## Results

```bash
Results



+-----------------------+------------+
|chunk                  |ner_label   |
+-----------------------+------------+
|Kentucky Fried Chicken |ORGANIZATION|
|Anglo-Egyptian War     |EVENT       |
|battle of Tell El Kebir|EVENT       |
|Egypt Medal            |OTHER       |
|Order of Medjidie      |OTHER       |
+-----------------------+------------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_fewnerd_100d_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|167.3 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter
- Finisher