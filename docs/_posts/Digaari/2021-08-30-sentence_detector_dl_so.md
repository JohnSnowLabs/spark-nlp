---
layout: model
title: Sentence Detection in Somali Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [open_source, sentence_detection, so]
task: Sentence Detection
language: so
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_so_3.2.0_3.0_1630321968392.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "so") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""Raadinta il weyn oo ka mid ah cutubyada akhriska Ingiriisiga? Waxaad timid meeshii saxda ahayd Sida laga soo xigtay daraasad dhowaan la sameeyay, caadadii wax -akhriska ee dhallinyarada maanta ayaa si degdeg ah hoos ugu dhacaysa. Waxay diiradda saari karin cutubka akhriska Ingiriisiga ee la siiyay wax ka badan dhowr ilbiriqsi! Sidoo kale, akhrintu waxay ahayd qayb muhiim ah oo ka mid ah dhammaan imtixaannada tartanka. Haddaba, sidee u hagaajin kartaa xirfadahaaga akhriska? Jawaabta su'aashan dhab ahaantii waa su'aal kale: Waa maxay isticmaalka xirfadaha akhriska? Ujeeddada ugu weyn ee wax -akhrisku waa 'macno samayn'.""")



```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "so")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("Raadinta il weyn oo ka mid ah cutubyada akhriska Ingiriisiga? Waxaad timid meeshii saxda ahayd Sida laga soo xigtay daraasad dhowaan la sameeyay, caadadii wax -akhriska ee dhallinyarada maanta ayaa si degdeg ah hoos ugu dhacaysa. Waxay diiradda saari karin cutubka akhriska Ingiriisiga ee la siiyay wax ka badan dhowr ilbiriqsi! Sidoo kale, akhrintu waxay ahayd qayb muhiim ah oo ka mid ah dhammaan imtixaannada tartanka. Haddaba, sidee u hagaajin kartaa xirfadahaaga akhriska? Jawaabta su'aashan dhab ahaantii waa su'aal kale: Waa maxay isticmaalka xirfadaha akhriska? Ujeeddada ugu weyn ee wax -akhrisku waa 'macno samayn'.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
nlu.load('so.sentence_detector').predict("Raadinta il weyn oo ka mid ah cutubyada akhriska Ingiriisiga? Waxaad timid meeshii saxda ahayd Sida laga soo xigtay daraasad dhowaan la sameeyay, caadadii wax -akhriska ee dhallinyarada maanta ayaa si degdeg ah hoos ugu dhacaysa. Waxay diiradda saari karin cutubka akhriska Ingiriisiga ee la siiyay wax ka badan dhowr ilbiriqsi! Sidoo kale, akhrintu waxay ahayd qayb muhiim ah oo ka mid ah dhammaan imtixaannada tartanka. Haddaba, sidee u hagaajin kartaa xirfadahaaga akhriska? Jawaabta su'aashan dhab ahaantii waa su'aal kale: Waa maxay isticmaalka xirfadaha akhriska? Ujeeddada ugu weyn ee wax -akhrisku waa 'macno samayn'.", output_level ='sentence')  
```
</div>

## Results

```bash
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                   |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[Raadinta il weyn oo ka mid ah cutubyada akhriska Ingiriisiga?]                                                                                                          |
|[Waxaad timid meeshii saxda ahayd Sida laga soo xigtay daraasad dhowaan la sameeyay, caadadii wax -akhriska ee dhallinyarada maanta ayaa si degdeg ah hoos ugu dhacaysa.]|
|[Waxay diiradda saari karin cutubka akhriska Ingiriisiga ee la siiyay wax ka badan dhowr ilbiriqsi!]                                                                     |
|[Sidoo kale, akhrintu waxay ahayd qayb muhiim ah oo ka mid ah dhammaan imtixaannada tartanka.]                                                                           |
|[Haddaba, sidee u hagaajin kartaa xirfadahaaga akhriska?]                                                                                                                |
|[Jawaabta su'aashan dhab ahaantii waa su'aal kale:]                                                                                                                      |
|[Waa maxay isticmaalka xirfadaha akhriska?]                                                                                                                              |
|[Ujeeddada ugu weyn ee wax -akhrisku waa 'macno samayn'.]                                                                                                                |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


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
|Language:|so|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```
