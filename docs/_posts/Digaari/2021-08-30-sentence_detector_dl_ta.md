---
layout: model
title: Sentence Detection in Tamil Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [ta, open_source, sentence_detection]
task: Sentence Detection
language: ta
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_ta_3.2.0_3.0_1630337465197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_ta_3.2.0_3.0_1630337465197.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "ta") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

sd_model.fullAnnotate("""ஆங்கில வாசிப்பு பத்திகளின் சிறந்த ஆதாரத்தைத் தேடுகிறீர்களா? நீங்கள் சரியான இடத்திற்கு வந்துவிட்டீர்கள். சமீபத்திய ஆய்வின்படி, இன்றைய இளைஞர்களிடம் படிக்கும் பழக்கம் வேகமாக குறைந்து வருகிறது. கொடுக்கப்பட்ட ஆங்கில வாசிப்பு பத்தியில் சில வினாடிகளுக்கு மேல் அவர்களால் கவனம் செலுத்த முடியாது! மேலும், அனைத்து போட்டித் தேர்வுகளிலும் வாசிப்பு ஒரு ஒருங்கிணைந்த பகுதியாகும். எனவே, உங்கள் வாசிப்புத் திறனை எவ்வாறு மேம்படுத்துவது? இந்த கேள்விக்கான பதில் உண்மையில் மற்றொரு கேள்வி: வாசிப்பு திறனின் பயன் என்ன? வாசிப்பின் முக்கிய நோக்கம் 'உணர்த்துவது'.""")

```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "ta")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val data = Seq("ஆங்கில வாசிப்பு பத்திகளின் சிறந்த ஆதாரத்தைத் தேடுகிறீர்களா? நீங்கள் சரியான இடத்திற்கு வந்துவிட்டீர்கள். சமீபத்திய ஆய்வின்படி, இன்றைய இளைஞர்களிடம் படிக்கும் பழக்கம் வேகமாக குறைந்து வருகிறது. கொடுக்கப்பட்ட ஆங்கில வாசிப்பு பத்தியில் சில வினாடிகளுக்கு மேல் அவர்களால் கவனம் செலுத்த முடியாது! மேலும், அனைத்து போட்டித் தேர்வுகளிலும் வாசிப்பு ஒரு ஒருங்கிணைந்த பகுதியாகும். எனவே, உங்கள் வாசிப்புத் திறனை எவ்வாறு மேம்படுத்துவது? இந்த கேள்விக்கான பதில் உண்மையில் மற்றொரு கேள்வி: வாசிப்பு திறனின் பயன் என்ன? வாசிப்பின் முக்கிய நோக்கம் 'உணர்த்துவது'.").toDF("text")

val result = pipeline.fit(data).transform(data)


```

{:.nlu-block}
```python
import nlu

nlu.load('ta.sentence_detector').predict("ஆங்கில வாசிப்பு பத்திகளின் சிறந்த ஆதாரத்தைத் தேடுகிறீர்களா? நீங்கள் சரியான இடத்திற்கு வந்துவிட்டீர்கள். சமீபத்திய ஆய்வின்படி, இன்றைய இளைஞர்களிடம் படிக்கும் பழக்கம் வேகமாக குறைந்து வருகிறது. கொடுக்கப்பட்ட ஆங்கில வாசிப்பு பத்தியில் சில வினாடிகளுக்கு மேல் அவர்களால் கவனம் செலுத்த முடியாது! மேலும், அனைத்து போட்டித் தேர்வுகளிலும் வாசிப்பு ஒரு ஒருங்கிணைந்த பகுதியாகும். எனவே, உங்கள் வாசிப்புத் திறனை எவ்வாறு மேம்படுத்துவது? இந்த கேள்விக்கான பதில் உண்மையில் மற்றொரு கேள்வி: வாசிப்பு திறனின் பயன் என்ன? வாசிப்பின் முக்கிய நோக்கம் 'உணர்த்துவது'.", output_level ='sentence')  
```
</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------+
|result                                                                                            |
+--------------------------------------------------------------------------------------------------+
|[ஆங்கில வாசிப்பு பத்திகளின் சிறந்த ஆதாரத்தைத் தேடுகிறீர்களா?]                                     |
|[நீங்கள் சரியான இடத்திற்கு வந்துவிட்டீர்கள்.]                                                     |
|[சமீபத்திய ஆய்வின்படி, இன்றைய இளைஞர்களிடம் படிக்கும் பழக்கம் வேகமாக குறைந்து வருகிறது.]           |
|[கொடுக்கப்பட்ட ஆங்கில வாசிப்பு பத்தியில் சில வினாடிகளுக்கு மேல் அவர்களால் கவனம் செலுத்த முடியாது!]|
|[மேலும், அனைத்து போட்டித் தேர்வுகளிலும் வாசிப்பு ஒரு ஒருங்கிணைந்த பகுதியாகும்.]                   |
|[எனவே, உங்கள் வாசிப்புத் திறனை எவ்வாறு மேம்படுத்துவது?]                                           |
|[இந்த கேள்விக்கான பதில் உண்மையில் மற்றொரு கேள்வி:]                                                |
|[வாசிப்பு திறனின் பயன் என்ன?]                                                                     |
|[வாசிப்பின் முக்கிய நோக்கம் 'உணர்த்துவது'.]                                                       |
+--------------------------------------------------------------------------------------------------+


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
|Language:|ta|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```