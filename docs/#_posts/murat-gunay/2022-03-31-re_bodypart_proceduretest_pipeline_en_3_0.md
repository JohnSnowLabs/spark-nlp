---
layout: model
title: RE Pipeline between Body Parts and Procedures
author: John Snow Labs
name: re_bodypart_proceduretest_pipeline
date: 2022-03-31
tags: [licensed, clinical, relation_extraction, body_part, procedures, en]
task: Relation Extraction
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


This pretrained pipeline is built on the top of [re_bodypart_proceduretest](https://nlp.johnsnowlabs.com/2021/01/18/re_bodypart_proceduretest_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_BODYPART_ENT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_bodypart_proceduretest_pipeline_en_3.4.1_3.0_1648733647318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("re_bodypart_proceduretest_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("re_bodypart_proceduretest_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.")
```
</div>


## Results


```bash
| index | relations | entity1                      | entity1_begin | entity1_end | chunk1 | entity2 | entity2_end | entity2_end | chunk2              | confidence |
|-------|-----------|------------------------------|---------------|-------------|--------|---------|-------------|-------------|---------------------|------------|
| 0     | 1         | External_body_part_or_region | 94            | 98          | chest  | Test    | 117         | 135         | portable ultrasound | 1.0        |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_bodypart_proceduretest_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|


## Included Models


- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- DependencyParserModel
- RelationExtractionModel
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg2NTUwMzU1NCwxMzk5NDk3Njk0LC0yMD
MyNjA5Mzg3XX0=
-->