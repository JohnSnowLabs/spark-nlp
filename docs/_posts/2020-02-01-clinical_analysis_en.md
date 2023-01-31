---
layout: model
title: Clinical Analysis
author: John Snow Labs
name: clinical_analysis
class: PipelineModel
language: en
repository: clinical/models
date: 2020-02-01
task: Pipeline Healthcare
edition: Healthcare NLP 2.4.0
spark_version: 2.4
tags: [clinical,licensed,pipeline,en]
supported: true
annotator: PipelineModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_analysis_en_2.4.0_2.4_1580600773378.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_analysis_en_2.4.0_2.4_1580600773378.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PretrainedPipeline("clinical_analysis","en","clinical/models")
```

```scala
val model = PipelineModel.pretrained("clinical_analysis","en","clinical/models")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|-------------------|
| Name:          | clinical_analysis |
| Type:   | PipelineModel     |
| Compatibility: | Spark NLP 2.4.0+             |
| License:       | Licensed          |
| Edition:       | Official        |
| Language:      | en                |


{:.h2_title}
## Data Source
