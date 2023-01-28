---
layout: model
title: Sentence Detection in Marathi Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [open_source, sentence_detection, mr]
task: Sentence Detection
language: mr
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_mr_3.2.0_3.0_1630319297311.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_mr_3.2.0_3.0_1630319297311.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "mr") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""इंग्रजी वाचन परिच्छेद एक उत्तम स्रोत शोधत आहात? आपण योग्य ठिकाणी आला आहात. नुकत्याच झालेल्या एका अभ्यासानुसार, आजच्या तरुणांमध्ये वाचनाची सवय झपाट्याने कमी होत आहे. ते दिलेल्या इंग्रजी वाचनाच्या परिच्छेदावर काही सेकंदांपेक्षा जास्त काळ लक्ष केंद्रित करू शकत नाहीत! तसेच, वाचन हा सर्व स्पर्धा परीक्षांचा अविभाज्य भाग होता आणि आहे. तर, तुम्ही तुमचे वाचन कौशल्य कसे सुधारता? या प्रश्नाचे उत्तर प्रत्यक्षात दुसरा प्रश्न आहे: वाचन कौशल्याचा उपयोग काय आहे? वाचनाचा मुख्य हेतू म्हणजे 'अर्थ काढणे'.""")

```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "mr")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))
val data = Seq("इंग्रजी वाचन परिच्छेद एक उत्तम स्रोत शोधत आहात? आपण योग्य ठिकाणी आला आहात. नुकत्याच झालेल्या एका अभ्यासानुसार, आजच्या तरुणांमध्ये वाचनाची सवय झपाट्याने कमी होत आहे. ते दिलेल्या इंग्रजी वाचनाच्या परिच्छेदावर काही सेकंदांपेक्षा जास्त काळ लक्ष केंद्रित करू शकत नाहीत! तसेच, वाचन हा सर्व स्पर्धा परीक्षांचा अविभाज्य भाग होता आणि आहे. तर, तुम्ही तुमचे वाचन कौशल्य कसे सुधारता? या प्रश्नाचे उत्तर प्रत्यक्षात दुसरा प्रश्न आहे: वाचन कौशल्याचा उपयोग काय आहे? वाचनाचा मुख्य हेतू म्हणजे 'अर्थ काढणे'.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
nlu.load('mr.sentence_detector').predict("इंग्रजी वाचन परिच्छेद एक उत्तम स्रोत शोधत आहात? आपण योग्य ठिकाणी आला आहात. नुकत्याच झालेल्या एका अभ्यासानुसार, आजच्या तरुणांमध्ये वाचनाची सवय झपाट्याने कमी होत आहे. ते दिलेल्या इंग्रजी वाचनाच्या परिच्छेदावर काही सेकंदांपेक्षा जास्त काळ लक्ष केंद्रित करू शकत नाहीत! तसेच, वाचन हा सर्व स्पर्धा परीक्षांचा अविभाज्य भाग होता आणि आहे. तर, तुम्ही तुमचे वाचन कौशल्य कसे सुधारता? या प्रश्नाचे उत्तर प्रत्यक्षात दुसरा प्रश्न आहे: वाचन कौशल्याचा उपयोग काय आहे? वाचनाचा मुख्य हेतू म्हणजे 'अर्थ काढणे'.", output_level ='sentence')  
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------+
|result                                                                                               |
+-----------------------------------------------------------------------------------------------------+
|[इंग्रजी वाचन परिच्छेद एक उत्तम स्रोत शोधत आहात?]						               |
|[आपण योग्य ठिकाणी आला आहात.]                                                                         |
|[नुकत्याच झालेल्या एका अभ्यासानुसार, आजच्या तरुणांमध्ये वाचनाची सवय झपाट्याने कमी होत आहे.]                           | 
|[ते दिलेल्या इंग्रजी वाचनाच्या परिच्छेदावर काही सेकंदांपेक्षा जास्त काळ लक्ष केंद्रित करू शकत नाहीत!]                          |
|[तसेच, वाचन हा सर्व स्पर्धा परीक्षांचा अविभाज्य भाग होता आणि आहे.]                                   		   |
|[तर, तुम्ही तुमचे वाचन कौशल्य कसे सुधारता?]                                                                  |
|[या प्रश्नाचे उत्तर प्रत्यक्षात दुसरा प्रश्न आहे:]                                                                     |
|[वाचन कौशल्याचा उपयोग काय आहे?]                                                                        |
|[वाचनाचा मुख्य हेतू म्हणजे 'अर्थ काढणे'.]                                                                    |
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
|Language:|mr|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```
