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
edition: Healthcare NLP 2.4.0
spark_version: 2.4
tags: [pipeline, clinical, licensed]
supported: true
annotator: PipelineModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This pipeline can be used to de-identify PHI information from medical texts. The PHI information will be obfuscated in the resulting text. 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_en_2.4.0_2.4_1580481115376.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


{:.h2_title}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

sample = """Name : Hendrickson, Ora, Record date: 2093-01-13, # 719435.
Dr. John Green, ID: 1231511863, IP 203.120.223.13.
He is a 60-year-old male was admitted to the Day Hospital for cystectomy on 01/13/93.
Patient's VIN : 1HGBH41JXMN109286, SSN #333-44-6666, Driver's license no:A334455B.
Phone (302) 786-5227, 0295 Keats Street, San Francisco, E-MAIL: smith@gmail.com."""

model = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")
result = deid_pipeline.annotate(sample)
```

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val sample = """Name : Hendrickson, Ora, Record date: 2093-01-13, # 719435.
Dr. John Green, ID: 1231511863, IP 203.120.223.13.
He is a 60-year-old male was admitted to the Day Hospital for cystectomy on 01/13/93.
Patient's VIN : 1HGBH41JXMN109286, SSN #333-44-6666, Driver's license no:A334455B.
Phone (302) 786-5227, 0295 Keats Street, San Francisco, E-MAIL: smith@gmail.com."""


val model = new PretrainedPipeline("clinical_deidentification", "en", "clinical/models")
val result = deid_pipeline.annotate(sample)
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



