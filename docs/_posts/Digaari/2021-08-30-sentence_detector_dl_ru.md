---
layout: model
title: Sentence Detection in Russian Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [ru, open_source, sentence_detection]
task: Sentence Detection
language: ru
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_ru_3.2.0_3.0_1630320562697.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

The sample code snippet may not contain all required fields of a pipeline. In this case, you can reach out a related colab notebook containing the end-to-end pipeline and more by clicking the "Open in Colab" link above.




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "ru") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""Ищете отличный источник абзацев для чтения на английском? Вы пришли в нужное место. Согласно недавнему исследованию, привычка к чтению у современной молодежи стремительно сокращается. Они не могут сосредоточиться на данном абзаце для чтения на английском дольше нескольких секунд! Кроме того, чтение было и остается неотъемлемой частью всех конкурсных экзаменов. Итак, как улучшить свои навыки чтения? Ответ на этот вопрос на самом деле представляет собой другой вопрос: какова польза от навыков чтения? Основная цель чтения - «понять смысл».""")


```
```scala
val documenter = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "ru")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("Ищете отличный источник абзацев для чтения на английском? Вы пришли в нужное место. Согласно недавнему исследованию, привычка к чтению у современной молодежи стремительно сокращается. Они не могут сосредоточиться на данном абзаце для чтения на английском дольше нескольких секунд! Кроме того, чтение было и остается неотъемлемой частью всех конкурсных экзаменов. Итак, как улучшить свои навыки чтения? Ответ на этот вопрос на самом деле представляет собой другой вопрос: какова польза от навыков чтения? Основная цель чтения - «понять смысл».").toDF("text")
val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python
nlu.load('ru.sentence_detector').predict("Ищете отличный источник абзацев для чтения на английском? Вы пришли в нужное место. Согласно недавнему исследованию, привычка к чтению у современной молодежи стремительно сокращается. Они не могут сосредоточиться на данном абзаце для чтения на английском дольше нескольких секунд! Кроме того, чтение было и остается неотъемлемой частью всех конкурсных экзаменов. Итак, как улучшить свои навыки чтения? Ответ на этот вопрос на самом деле представляет собой другой вопрос: какова польза от навыков чтения? Основная цель чтения - «понять смысл».", output_level ='sentence')  
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------+
|result                                                                                               |
+-----------------------------------------------------------------------------------------------------+
|[Ищете отличный источник абзацев для чтения на английском?]                                          |
|[Вы пришли в нужное место.]                                                                          |
|[Согласно недавнему исследованию, привычка к чтению у современной молодежи стремительно сокращается.]|
|[Они не могут сосредоточиться на данном абзаце для чтения на английском дольше нескольких секунд!]   |
|[Кроме того, чтение было и остается неотъемлемой частью всех конкурсных экзаменов.]                  |
|[Итак, как улучшить свои навыки чтения?]                                                             |
|[Ответ на этот вопрос на самом деле представляет собой другой вопрос:]                               |
|[какова польза от навыков чтения?]                                                                   |
|[Основная цель чтения - «понять смысл».]                                                             |
+-----------------------------------------------------------------------------------------------------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|ru|

## Benchmarking

```bash
Accuracy:      0.98
Recall:        1.00
Precision:     0.96
F1:            0.98

```
