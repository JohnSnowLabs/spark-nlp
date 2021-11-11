---
layout: model
title: Clinical Deidentification
author: John Snow Labs
name: clinical_deidentification
class: PipelineModel
language: en
repository: clinical/models
date: 2020-01-31
task: [De-identification, Pipeline Healthcare]
edition: Spark NLP for Healthcare 2.4.0
tags: [pipeline, clinical, licensed]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This pipeline can be used to de-identify PHI information from medical texts. The PHI information will be obfuscated in the resulting text. 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_en_2.4.0_2.4_1580481115376.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("clinical_deidentification","en","clinical/models")
```

```scala
val model = PretrainedPipeline("clinical_deidentification","en","clinical/models")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------------|
| Name:          | clinical_deidentification |
| Type:   | PipelineModel             |
| Compatibility: | Spark NLP 2.4.0+                     |
| License:       | Licensed                  |
| Edition:       | Official                |
| Language:      | en                        |


{:.h2_title}
## Data Source
