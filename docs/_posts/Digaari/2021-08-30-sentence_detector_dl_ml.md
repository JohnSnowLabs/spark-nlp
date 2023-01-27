---
layout: model
title: Sentence Detection in Malayalam Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [ml, sentence_detection, open_source]
task: Sentence Detection
language: ml
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_ml_3.2.0_3.0_1630336657068.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_ml_3.2.0_3.0_1630336657068.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "ml") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

sd_model.fullAnnotate("""ഇംഗ്ലീഷ് വായിക്കുന്ന ഖണ്ഡികകളുടെ മികച്ച ഉറവിടം തേടുകയാണോ? നിങ്ങൾ ശരിയായ സ്ഥലത്ത് എത്തിയിരിക്കുന്നു. അടുത്തിടെ നടത്തിയ ഒരു പഠനമനുസരിച്ച്, ഇന്നത്തെ യുവാക്കളിൽ വായനാശീലം അതിവേഗം കുറഞ്ഞുവരികയാണ്. ഒരു നിശ്ചിത സെക്കൻഡിൽ കൂടുതൽ ഒരു ഇംഗ്ലീഷ് വായന ഖണ്ഡികയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കാൻ അവർക്ക് കഴിയില്ല! കൂടാതെ, വായന എല്ലാ മത്സര പരീക്ഷകളുടെയും അവിഭാജ്യ ഘടകമായിരുന്നു. അതിനാൽ, നിങ്ങളുടെ വായനാ കഴിവുകൾ എങ്ങനെ മെച്ചപ്പെടുത്താം? ഈ ചോദ്യത്തിനുള്ള ഉത്തരം യഥാർത്ഥത്തിൽ മറ്റൊരു ചോദ്യമാണ്: വായനാ വൈദഗ്ധ്യത്തിന്റെ ഉപയോഗം എന്താണ്? വായനയുടെ പ്രധാന ലക്ഷ്യം 'അർത്ഥവത്താക്കുക' എന്നതാണ്.""")

```
```scala
val documenter = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "ml")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val data = Seq("ഇംഗ്ലീഷ് വായിക്കുന്ന ഖണ്ഡികകളുടെ മികച്ച ഉറവിടം തേടുകയാണോ? നിങ്ങൾ ശരിയായ സ്ഥലത്ത് എത്തിയിരിക്കുന്നു. അടുത്തിടെ നടത്തിയ ഒരു പഠനമനുസരിച്ച്, ഇന്നത്തെ യുവാക്കളിൽ വായനാശീലം അതിവേഗം കുറഞ്ഞുവരികയാണ്. ഒരു നിശ്ചിത സെക്കൻഡിൽ കൂടുതൽ ഒരു ഇംഗ്ലീഷ് വായന ഖണ്ഡികയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കാൻ അവർക്ക് കഴിയില്ല! കൂടാതെ, വായന എല്ലാ മത്സര പരീക്ഷകളുടെയും അവിഭാജ്യ ഘടകമായിരുന്നു. അതിനാൽ, നിങ്ങളുടെ വായനാ കഴിവുകൾ എങ്ങനെ മെച്ചപ്പെടുത്താം? ഈ ചോദ്യത്തിനുള്ള ഉത്തരം യഥാർത്ഥത്തിൽ മറ്റൊരു ചോദ്യമാണ്: വായനാ വൈദഗ്ധ്യത്തിന്റെ ഉപയോഗം എന്താണ്? വായനയുടെ പ്രധാന ലക്ഷ്യം 'അർത്ഥവത്താക്കുക' എന്നതാണ്.").toDF("text")

val result = pipeline.fit(data).transform(data)

```

{:.nlu-block}
```python
import nlu

nlu.load('ml.sentence_detector').predict("ഇംഗ്ലീഷ് വായിക്കുന്ന ഖണ്ഡികകളുടെ മികച്ച ഉറവിടം തേടുകയാണോ? നിങ്ങൾ ശരിയായ സ്ഥലത്ത് എത്തിയിരിക്കുന്നു. അടുത്തിടെ നടത്തിയ ഒരു പഠനമനുസരിച്ച്, ഇന്നത്തെ യുവാക്കളിൽ വായനാശീലം അതിവേഗം കുറഞ്ഞുവരികയാണ്. ഒരു നിശ്ചിത സെക്കൻഡിൽ കൂടുതൽ ഒരു ഇംഗ്ലീഷ് വായന ഖണ്ഡികയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കാൻ അവർക്ക് കഴിയില്ല! കൂടാതെ, വായന എല്ലാ മത്സര പരീക്ഷകളുടെയും അവിഭാജ്യ ഘടകമായിരുന്നു. അതിനാൽ, നിങ്ങളുടെ വായനാ കഴിവുകൾ എങ്ങനെ മെച്ചപ്പെടുത്താം? ഈ ചോദ്യത്തിനുള്ള ഉത്തരം യഥാർത്ഥത്തിൽ മറ്റൊരു ചോദ്യമാണ്: വായനാ വൈദഗ്ധ്യത്തിന്റെ ഉപയോഗം എന്താണ്? വായനയുടെ പ്രധാന ലക്ഷ്യം 'അർത്ഥവത്താക്കുക' എന്നതാണ്.", output_level ='sentence')  
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+
|result                                                                                              |
+----------------------------------------------------------------------------------------------------+
|[ഇംഗ്ലീഷ് വായിക്കുന്ന ഖണ്ഡികകളുടെ മികച്ച ഉറവിടം തേടുകയാണോ?]                                         |
|[നിങ്ങൾ ശരിയായ സ്ഥലത്ത് എത്തിയിരിക്കുന്നു.]                                                         |
|[അടുത്തിടെ നടത്തിയ ഒരു പഠനമനുസരിച്ച്, ഇന്നത്തെ യുവാക്കളിൽ വായനാശീലം അതിവേഗം കുറഞ്ഞുവരികയാണ്.]       |
|[ഒരു നിശ്ചിത സെക്കൻഡിൽ കൂടുതൽ ഒരു ഇംഗ്ലീഷ് വായന ഖണ്ഡികയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കാൻ അവർക്ക് കഴിയില്ല!]|
|[കൂടാതെ, വായന എല്ലാ മത്സര പരീക്ഷകളുടെയും അവിഭാജ്യ ഘടകമായിരുന്നു.]                                   |
|[അതിനാൽ, നിങ്ങളുടെ വായനാ കഴിവുകൾ എങ്ങനെ മെച്ചപ്പെടുത്താം?]                                          |
|[ഈ ചോദ്യത്തിനുള്ള ഉത്തരം യഥാർത്ഥത്തിൽ മറ്റൊരു ചോദ്യമാണ്:]                                           |
|[വായനാ വൈദഗ്ധ്യത്തിന്റെ ഉപയോഗം എന്താണ്?]                                                            |
|[വായനയുടെ പ്രധാന ലക്ഷ്യം 'അർത്ഥവത്താക്കുക' എന്നതാണ്.]                                               |
+----------------------------------------------------------------------------------------------------+


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
|Language:|ml|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```