---
layout: model
title: Sentence Detection in Ukrainian Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [open_source, uk, sentence_detection]
task: Sentence Detection
language: uk
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_uk_3.2.0_3.0_1630322414306.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_uk_3.2.0_3.0_1630322414306.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "uk") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""Шукаєте чудове джерело англійського читання абзаців? Ви потрапили в потрібне місце. Згідно з останнім дослідженням, звичка читати у сучасної молоді стрімко знижується. Вони не можуть зосередитися на даному абзаці читання англійською мовою більше кількох секунд! Крім того, читання було і є невід’ємною частиною всіх конкурсних іспитів. Отже, як покращити свої навички читання? Відповідь на це питання насправді інше питання: Яка користь від навичок читання? Основна мета читання - «мати сенс».""")



```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "uk")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("Шукаєте чудове джерело англійського читання абзаців? Ви потрапили в потрібне місце. Згідно з останнім дослідженням, звичка читати у сучасної молоді стрімко знижується. Вони не можуть зосередитися на даному абзаці читання англійською мовою більше кількох секунд! Крім того, читання було і є невід’ємною частиною всіх конкурсних іспитів. Отже, як покращити свої навички читання? Відповідь на це питання насправді інше питання: Яка користь від навичок читання? Основна мета читання - «мати сенс».").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
nlu.load('uk.sentence_detector').predict('Шукаєте чудове джерело англійського читання абзаців? Ви потрапили в потрібне місце. Згідно з останнім дослідженням, звичка читати у сучасної молоді стрімко знижується. Вони не можуть зосередитися на даному абзаці читання англійською мовою більше кількох секунд! Крім того, читання було і є невід’ємною частиною всіх конкурсних іспитів. Отже, як покращити свої навички читання? Відповідь на це питання насправді інше питання: Яка користь від навичок читання? Основна мета читання - «мати сенс».', output_level ='sentence')  
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------+
|result                                                                                         |
+-----------------------------------------------------------------------------------------------+
|[Шукаєте чудове джерело англійського читання абзаців?]                                         |
|[Ви потрапили в потрібне місце.]                                                               |
|[Згідно з останнім дослідженням, звичка читати у сучасної молоді стрімко знижується.]          |
|[Вони не можуть зосередитися на даному абзаці читання англійською мовою більше кількох секунд!]|
|[Крім того, читання було і є невід’ємною частиною всіх конкурсних іспитів.]                    |
|[Отже, як покращити свої навички читання?]                                                     |
|[Відповідь на це питання насправді інше питання:]                                              |
|[Яка користь від навичок читання?]                                                             |
|[Основна мета читання - «мати сенс».]                                                          |
+-----------------------------------------------------------------------------------------------+


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
|Language:|uk|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```
