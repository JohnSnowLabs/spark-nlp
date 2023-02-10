---
layout: model
title: Pipeline to Extract Neurologic Deficits Related to Stroke Scale (NIHSS)
author: John Snow Labs
name: ner_nihss_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
task: [Named Entity Recognition, Pipeline Healthcare]
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This pretrained pipeline is built on the top of [ner_nihss](https://nlp.johnsnowlabs.com/2021/11/15/ner_nihss_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_NIHSS/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_NIHSS.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_nihss_pipeline_en_3.4.1_3.0_1647871076449.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_nihss_pipeline_en_3.4.1_3.0_1647871076449.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_nihss_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_nihss_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently.")
```
</div>


## Results


```bash
|    | chunk              | entity                   |
|---:|:-------------------|:-------------------------|
|  0 | NIH stroke scale   | NIHSS                    |
|  1 | 23 to 24           | Measurement              |
|  2 | one                | Measurement              |
|  3 | consciousness      | 1a_LOC                   |
|  4 | two                | Measurement              |
|  5 | month and year     | 1b_LOCQuestions          |
|  6 | two                | Measurement              |
|  7 | eye / grip         | 1c_LOCCommands           |
|  8 | one                | Measurement              |
|  9 | two                | Measurement              |
| 10 | gaze               | 2_BestGaze               |
| 11 | two                | Measurement              |
| 12 | face               | 4_FacialPalsy            |
| 13 | eight              | Measurement              |
| 14 | one                | Measurement              |
| 15 | limited ataxia     | 7_LimbAtaxia             |
| 16 | one to two         | Measurement              |
| 17 | sensory            | 8_Sensory                |
| 18 | three              | Measurement              |
| 19 | best language      | 9_BestLanguage           |
| 20 | two                | Measurement              |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_nihss_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|


## Included Models


- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTExMzEyNDEzNSwxNjIxNzkwMjM2XX0=
-->