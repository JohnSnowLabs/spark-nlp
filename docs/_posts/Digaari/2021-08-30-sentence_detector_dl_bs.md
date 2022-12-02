---
layout: model
title: Sentence Detection in Bosnian Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [bs, sentence_detection, open_source]
task: Sentence Detection
language: bs
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_bs_3.2.0_3.0_1630317779410.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "bs") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""Tražite sjajan izvor čitanja odlomaka na engleskom? Došli ste na pravo mjesto. Prema nedavnom istraživanju, navika čitanja u današnjoj mladosti brzo se smanjuje. Ne mogu se usredotočiti na dati odlomak za čitanje engleskog jezika duže od nekoliko sekundi! Takođe, čitanje je bilo i jeste sastavni dio svih takmičarskih ispita. Dakle, kako poboljšati svoje vještine čitanja? Odgovor na ovo pitanje zapravo je drugo pitanje: Kakva je korist od vještine čitanja? Glavna svrha čitanja je 'imati smisla'.""")


```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "bs")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("Tražite sjajan izvor čitanja odlomaka na engleskom? Došli ste na pravo mjesto. Prema nedavnom istraživanju, navika čitanja u današnjoj mladosti brzo se smanjuje. Ne mogu se usredotočiti na dati odlomak za čitanje engleskog jezika duže od nekoliko sekundi! Takođe, čitanje je bilo i jeste sastavni dio svih takmičarskih ispita. Dakle, kako poboljšati svoje vještine čitanja? Odgovor na ovo pitanje zapravo je drugo pitanje: Kakva je korist od vještine čitanja? Glavna svrha čitanja je 'imati smisla'.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
nlu.load('bs.sentence_detector').predict("Tražite sjajan izvor čitanja odlomaka na engleskom? Došli ste na pravo mjesto. Prema nedavnom istraživanju, navika čitanja u današnjoj mladosti brzo se smanjuje. Ne mogu se usredotočiti na dati odlomak za čitanje engleskog jezika duže od nekoliko sekundi! Takođe, čitanje je bilo i jeste sastavni dio svih takmičarskih ispita. Dakle, kako poboljšati svoje vještine čitanja? Odgovor na ovo pitanje zapravo je drugo pitanje: Kakva je korist od vještine čitanja? Glavna svrha čitanja je 'imati smisla'.", output_level ='sentence')  
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------+
|result                                                                                         |
+-----------------------------------------------------------------------------------------------+
|[Tražite sjajan izvor čitanja odlomaka na engleskom?]                                          |
|[Došli ste na pravo mjesto.]                                                                   |
|[Prema nedavnom istraživanju, navika čitanja u današnjoj mladosti brzo se smanjuje.]           |
|[Ne mogu se usredotočiti na dati odlomak za čitanje engleskog jezika duže od nekoliko sekundi!]|
|[Takođe, čitanje je bilo i jeste sastavni dio svih takmičarskih ispita.]                       |
|[Dakle, kako poboljšati svoje vještine čitanja?]                                               |
|[Odgovor na ovo pitanje zapravo je drugo pitanje:]                                             |
|[Kakva je korist od vještine čitanja?]                                                         |
|[Glavna svrha čitanja je 'imati smisla'.]                                                      |
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
|Language:|bs|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```