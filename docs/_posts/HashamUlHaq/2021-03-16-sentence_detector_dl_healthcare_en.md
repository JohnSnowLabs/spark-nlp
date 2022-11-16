---
layout: model
title: Split Sentences in Healthcare Texts
author: John Snow Labs
name: sentence_detector_dl_healthcare
date: 2021-03-16
tags: [en, sentence_detection, licensed, clinical]
task: Sentence Detection
language: en
edition: Healthcare NLP 2.7.0
spark_version: 2.4
supported: true
annotator: SentenceDetectorDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/SENTENCE_DETECTOR_HC/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/20.SentenceDetectorDL_Healthcare.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sentence_detector_dl_healthcare_en_2.7.0_2.4_1615880554391.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

result = sd_model.fullAnnotate("""He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv. LASIX CHANGED TO 40 PO BID WHICH IS SAME AS HE TAKES AT HOME - RECEIVED 40 PO IN AM - 700CC U/O TOTAL FOR FLUID NEGATIVE ~ 600 THUS FAR TODAY, ~ 600 NEG LOS.""")

```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv. LASIX CHANGED TO 40 PO BID WHICH IS SAME AS HE TAKES AT HOME - RECEIVED 40 PO IN AM - 700CC U/O TOTAL FOR FLUID NEGATIVE ~ 600 THUS FAR TODAY, ~ 600 NEG LOS.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.detect_sentence.clinical").predict("""He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv. LASIX CHANGED TO 40 PO BID WHICH IS SAME AS HE TAKES AT HOME - RECEIVED 40 PO IN AM - 700CC U/O TOTAL FOR FLUID NEGATIVE ~ 600 THUS FAR TODAY, ~ 600 NEG LOS.""")
```

</div>

## Results

```bash
|    | sentence                                                                                                                                                                                                               |
|---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety. |
|  1 | Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv.                                                                                                                                                   |
|  2 | LASIX CHANGED TO 40 PO BID WHICH IS SAME AS HE TAKES AT HOME - RECEIVED 40 PO IN AM - 700CC U/O TOTAL FOR FLUID NEGATIVE ~ 600 THUS FAR TODAY, ~ 600 NEG LOS.                                                          |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl_healthcare|
|Compatibility:|Healthcare NLP 2.7.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|en|

## Data Source

Healthcare SDDL model is trained on in-house domain specific data.

## Benchmarking


```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```

