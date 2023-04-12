---
layout: model
title: Sentence Detection in Kannada Text
author: John Snow Labs
name: sentence_detector_dl
date: 2021-08-30
tags: [kn, open_source, sentence_detection]
task: Sentence Detection
language: kn
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_kn_3.2.0_3.0_1630336398052.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_kn_3.2.0_3.0_1630336398052.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel\
.pretrained("sentence_detector_dl", "kn") \
.setInputCols(["document"]) \
.setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

sd_model.fullAnnotate("""ಇಂಗ್ಲಿಷ್ ಓದುವ ಪ್ಯಾರಾಗಳ ಉತ್ತಮ ಮೂಲವನ್ನು ಹುಡುಕುತ್ತಿರುವಿರಾ? ನೀವು ಸರಿಯಾದ ಸ್ಥಳಕ್ಕೆ ಬಂದಿದ್ದೀರಿ. ಇತ್ತೀಚಿನ ಅಧ್ಯಯನದ ಪ್ರಕಾರ, ಇಂದಿನ ಯುವಜನರಲ್ಲಿ ಓದುವ ಅಭ್ಯಾಸವು ವೇಗವಾಗಿ ಕಡಿಮೆಯಾಗುತ್ತಿದೆ. ಅವರು ಕೆಲವು ಸೆಕೆಂಡುಗಳಿಗಿಂತ ಹೆಚ್ಚು ಕಾಲ ಆಂಗ್ಲ ಓದುವ ಪ್ಯಾರಾಗ್ರಾಫ್ ಮೇಲೆ ಕೇಂದ್ರೀಕರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ! ಅಲ್ಲದೆ, ಓದುವುದು ಎಲ್ಲಾ ಸ್ಪರ್ಧಾತ್ಮಕ ಪರೀಕ್ಷೆಗಳ ಅವಿಭಾಜ್ಯ ಅಂಗವಾಗಿತ್ತು. ಹಾಗಾದರೆ, ನಿಮ್ಮ ಓದುವ ಕೌಶಲ್ಯವನ್ನು ನೀವು ಹೇಗೆ ಸುಧಾರಿಸುತ್ತೀರಿ? ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರವು ವಾಸ್ತವವಾಗಿ ಇನ್ನೊಂದು ಪ್ರಶ್ನೆಯಾಗಿದೆ: ಓದುವ ಕೌಶಲ್ಯದ ಉಪಯೋಗವೇನು? ಓದುವ ಮುಖ್ಯ ಉದ್ದೇಶ 'ಅರ್ಥ ಮಾಡಿಕೊಳ್ಳುವುದು'.""")
```
```scala
val documenter = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "kn")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(documenter, model))

val data = Seq("ಇಂಗ್ಲಿಷ್ ಓದುವ ಪ್ಯಾರಾಗಳ ಉತ್ತಮ ಮೂಲವನ್ನು ಹುಡುಕುತ್ತಿರುವಿರಾ? ನೀವು ಸರಿಯಾದ ಸ್ಥಳಕ್ಕೆ ಬಂದಿದ್ದೀರಿ. ಇತ್ತೀಚಿನ ಅಧ್ಯಯನದ ಪ್ರಕಾರ, ಇಂದಿನ ಯುವಜನರಲ್ಲಿ ಓದುವ ಅಭ್ಯಾಸವು ವೇಗವಾಗಿ ಕಡಿಮೆಯಾಗುತ್ತಿದೆ. ಅವರು ಕೆಲವು ಸೆಕೆಂಡುಗಳಿಗಿಂತ ಹೆಚ್ಚು ಕಾಲ ಆಂಗ್ಲ ಓದುವ ಪ್ಯಾರಾಗ್ರಾಫ್ ಮೇಲೆ ಕೇಂದ್ರೀಕರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ! ಅಲ್ಲದೆ, ಓದುವುದು ಎಲ್ಲಾ ಸ್ಪರ್ಧಾತ್ಮಕ ಪರೀಕ್ಷೆಗಳ ಅವಿಭಾಜ್ಯ ಅಂಗವಾಗಿತ್ತು. ಹಾಗಾದರೆ, ನಿಮ್ಮ ಓದುವ ಕೌಶಲ್ಯವನ್ನು ನೀವು ಹೇಗೆ ಸುಧಾರಿಸುತ್ತೀರಿ? ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರವು ವಾಸ್ತವವಾಗಿ ಇನ್ನೊಂದು ಪ್ರಶ್ನೆಯಾಗಿದೆ: ಓದುವ ಕೌಶಲ್ಯದ ಉಪಯೋಗವೇನು? ಓದುವ ಮುಖ್ಯ ಉದ್ದೇಶ 'ಅರ್ಥ ಮಾಡಿಕೊಳ್ಳುವುದು'.").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

nlu.load('kn.sentence_detector').predict("ಇಂಗ್ಲಿಷ್ ಓದುವ ಪ್ಯಾರಾಗಳ ಉತ್ತಮ ಮೂಲವನ್ನು ಹುಡುಕುತ್ತಿರುವಿರಾ? ನೀವು ಸರಿಯಾದ ಸ್ಥಳಕ್ಕೆ ಬಂದಿದ್ದೀರಿ. ಇತ್ತೀಚಿನ ಅಧ್ಯಯನದ ಪ್ರಕಾರ, ಇಂದಿನ ಯುವಜನರಲ್ಲಿ ಓದುವ ಅಭ್ಯಾಸವು ವೇಗವಾಗಿ ಕಡಿಮೆಯಾಗುತ್ತಿದೆ. ಅವರು ಕೆಲವು ಸೆಕೆಂಡುಗಳಿಗಿಂತ ಹೆಚ್ಚು ಕಾಲ ಆಂಗ್ಲ ಓದುವ ಪ್ಯಾರಾಗ್ರಾಫ್ ಮೇಲೆ ಕೇಂದ್ರೀಕರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ! ಅಲ್ಲದೆ, ಓದುವುದು ಎಲ್ಲಾ ಸ್ಪರ್ಧಾತ್ಮಕ ಪರೀಕ್ಷೆಗಳ ಅವಿಭಾಜ್ಯ ಅಂಗವಾಗಿತ್ತು. ಹಾಗಾದರೆ, ನಿಮ್ಮ ಓದುವ ಕೌಶಲ್ಯವನ್ನು ನೀವು ಹೇಗೆ ಸುಧಾರಿಸುತ್ತೀರಿ? ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರವು ವಾಸ್ತವವಾಗಿ ಇನ್ನೊಂದು ಪ್ರಶ್ನೆಯಾಗಿದೆ: ಓದುವ ಕೌಶಲ್ಯದ ಉಪಯೋಗವೇನು? ಓದುವ ಮುಖ್ಯ ಉದ್ದೇಶ 'ಅರ್ಥ ಮಾಡಿಕೊಳ್ಳುವುದು'.", output_level ='sentence')  

```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------+
|result                                                                                       |
+---------------------------------------------------------------------------------------------+
|[ಇಂಗ್ಲಿಷ್ ಓದುವ ಪ್ಯಾರಾಗಳ ಉತ್ತಮ ಮೂಲವನ್ನು ಹುಡುಕುತ್ತಿರುವಿರಾ?]                                    				      |
|[ನೀವು ಸರಿಯಾದ ಸ್ಥಳಕ್ಕೆ ಬಂದಿದ್ದೀರಿ.]                                                           						      |
|[ಇತ್ತೀಚಿನ ಅಧ್ಯಯನದ ಪ್ರಕಾರ, ಇಂದಿನ ಯುವಜನರಲ್ಲಿ ಓದುವ ಅಭ್ಯಾಸವು ವೇಗವಾಗಿ ಕಡಿಮೆಯಾಗುತ್ತಿದೆ.]          			      |
|[ಅವರು ಕೆಲವು ಸೆಕೆಂಡುಗಳಿಗಿಂತ ಹೆಚ್ಚು ಕಾಲ ಆಂಗ್ಲ ಓದುವ ಪ್ಯಾರಾಗ್ರಾಫ್ ಮೇಲೆ ಕೇಂದ್ರೀಕರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ!]			      |
|[ಅಲ್ಲದೆ, ಓದುವುದು ಎಲ್ಲಾ ಸ್ಪರ್ಧಾತ್ಮಕ ಪರೀಕ್ಷೆಗಳ ಅವಿಭಾಜ್ಯ ಅಂಗವಾಗಿತ್ತು.]                        					      |
|[ಹಾಗಾದರೆ, ನಿಮ್ಮ ಓದುವ ಕೌಶಲ್ಯವನ್ನು ನೀವು ಹೇಗೆ ಸುಧಾರಿಸುತ್ತೀರಿ?]                             					      |
|[ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರವು ವಾಸ್ತವವಾಗಿ ಇನ್ನೊಂದು ಪ್ರಶ್ನೆಯಾಗಿದೆ:]                               						      |
|[ಓದುವ ಕೌಶಲ್ಯದ ಉಪಯೋಗವೇನು?]                                                               						      |
|[ಓದುವ ಮುಖ್ಯ ಉದ್ದೇಶ 'ಅರ್ಥ ಮಾಡಿಕೊಳ್ಳುವುದು'.]                                             					      |
+---------------------------------------------------------------------------------------------+

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
|Language:|kn|

## Benchmarking

```bash
label  Accuracy  Recall   Prec   F1  
0      0.98      1.00     0.96   0.98
```