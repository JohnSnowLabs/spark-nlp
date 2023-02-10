---
layout: model
title: Sentence Detection in Gujrati Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [gu, sentence_detection, open_source]
task: Sentence Detection
language: gu
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_gu_3.2.0_3.0_1630336149356.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_gu_3.2.0_3.0_1630336149356.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "gu") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

sd_model.fullAnnotate("""ઇંગલિશ વાંચન ફકરા એક મહાન સ્ત્રોત માટે શોધી રહ્યાં છો? તમે યોગ્ય જગ્યાએ આવ્યા છો. તાજેતરના એક અભ્યાસ મુજબ આજના યુવાનોમાં વાંચવાની ટેવ ઝડપથી ઘટી રહી છે. તેઓ આપેલ અંગ્રેજી વાંચન ફકરા પર થોડી સેકંડથી વધુ સમય સુધી ધ્યાન કેન્દ્રિત કરી શકતા નથી! ઉપરાંત, વાંચન તમામ સ્પર્ધાત્મક પરીક્ષાઓનો અભિન્ન ભાગ હતો અને છે. તો, તમે તમારી વાંચન કુશળતા કેવી રીતે સુધારી શકો છો? આ પ્રશ્નનો જવાબ વાસ્તવમાં બીજો પ્રશ્ન છે: વાંચન કુશળતાનો ઉપયોગ શું છે? વાંચનનો મુખ્ય હેતુ 'અર્થપૂર્ણ બનાવવાનો' છે.""")

```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "gu")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val data = Seq("ઇંગલિશ વાંચન ફકરા એક મહાન સ્ત્રોત માટે શોધી રહ્યાં છો? તમે યોગ્ય જગ્યાએ આવ્યા છો. તાજેતરના એક અભ્યાસ મુજબ આજના યુવાનોમાં વાંચવાની ટેવ ઝડપથી ઘટી રહી છે. તેઓ આપેલ અંગ્રેજી વાંચન ફકરા પર થોડી સેકંડથી વધુ સમય સુધી ધ્યાન કેન્દ્રિત કરી શકતા નથી! ઉપરાંત, વાંચન તમામ સ્પર્ધાત્મક પરીક્ષાઓનો અભિન્ન ભાગ હતો અને છે. તો, તમે તમારી વાંચન કુશળતા કેવી રીતે સુધારી શકો છો? આ પ્રશ્નનો જવાબ વાસ્તવમાં બીજો પ્રશ્ન છે: વાંચન કુશળતાનો ઉપયોગ શું છે? વાંચનનો મુખ્ય હેતુ 'અર્થપૂર્ણ બનાવવાનો' છે.").toDF("text")

val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python
import nlu

nlu.load('gu.sentence_detector').predict("अंग्रेजी पढ्ने अनुच्छेद को एक महान स्रोत को लागी हेर्दै हुनुहुन्छ? तपाइँ सही ठाउँमा आउनुभएको छ. हालै गरिएको एक अध्ययन अनुसार आजको युवाहरुमा पढ्ने बानी छिटोछिटो घट्दै गएको छ. उनीहरु केहि सेकेन्ड भन्दा बढी को लागी एक दिईएको अंग्रेजी पढ्ने अनुच्छेद मा ध्यान केन्द्रित गर्न सक्दैनन्! साथै, पठन थियो र सबै प्रतियोगी परीक्षा को एक अभिन्न हिस्सा हो। त्यसोभए, तपाइँ तपाइँको पठन कौशल कसरी सुधार गर्नुहुन्छ? यो प्रश्न को जवाफ वास्तव मा अर्को प्रश्न हो: पढ्ने कौशल को उपयोग के हो? पढ्न को मुख्य उद्देश्य 'भावना बनाउन' हो.", output_level ='sentence')  
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------+
|result                                                                                   |
+-----------------------------------------------------------------------------------------+
|[ઇંગલિશ વાંચન ફકરા એક મહાન સ્ત્રોત માટે શોધી રહ્યાં છો?]                                 						  |
|[તમે યોગ્ય જગ્યાએ આવ્યા છો.]                                                            						  |
|[તાજેતરના એક અભ્યાસ મુજબ આજના યુવાનોમાં વાંચવાની ટેવ ઝડપથી ઘટી રહી છે.]             				     	  |
|[તેઓ આપેલ અંગ્રેજી વાંચન ફકરા પર થોડી સેકંડથી વધુ સમય સુધી ધ્યાન કેન્દ્રિત કરી શકતા નથી!]					  |
|[ઉપરાંત, વાંચન તમામ સ્પર્ધાત્મક પરીક્ષાઓનો અભિન્ન ભાગ હતો અને છે.]                       					  |
|[તો, તમે તમારી વાંચન કુશળતા કેવી રીતે સુધારી શકો છો?]                                    						  |
|[આ પ્રશ્નનો જવાબ વાસ્તવમાં બીજો પ્રશ્ન છે:]                                              						  |
|[વાંચન કુશળતાનો ઉપયોગ શું છે?]                                                           						  |
|[વાંચનનો મુખ્ય હેતુ 'અર્થપૂર્ણ બનાવવાનો' છે.]                                            						  |
+-----------------------------------------------------------------------------------------+
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
|Language:|gu|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```