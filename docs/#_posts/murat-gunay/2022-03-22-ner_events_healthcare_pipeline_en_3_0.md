---
layout: model
title: Pipeline to Detect Clinical Events
author: John Snow Labs
name: ner_events_healthcare_pipeline
date: 2022-03-22
tags: [licensed, ner, clinical, en]
task: Named Entity Recognition
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


This pretrained pipeline is built on the top of [ner_events_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_healthcare_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_EVENTS_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_healthcare_pipeline_en_3.4.1_3.0_1647943997404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_events_healthcare_pipeline", "en", "clinical/models")


pipeline.fullAnnotate("The patient presented to the emergency room last evening")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_events_healthcare_pipeline", "en", "clinical/models")


pipeline.fullAnnotate("The patient presented to the emergency room last evening")
```
</div>


## Results


```bash
+------------------+-------------+
|chunks            |entities     |
+------------------+-------------+
|presented         |EVIDENTIAL   |
|the emergency room|CLINICAL_DEPT|
|last evening      |DATE         |
+------------------+-------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_events_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.6 MB|


## Included Models


- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzk1OTQzMzE2LDMxOTMzNjUyNV19
-->