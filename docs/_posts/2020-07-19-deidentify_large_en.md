---
layout: model
title: Deidentify Large
author: John Snow Labs
name: deidentify_large
class: DeIdentificationModel
language: en
repository: clinical/models
date: 2020-07-19
tags: [clinical,deidentification,regex,large,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Anonymization and DeIdentification model based on outputs from DeId NERs and Replacement Dictionaries  
Deidentify (Large) is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing "2020,06,04" with "<DATE>"). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

{:.h2_title}
## Prediction Domain
Contact, Location, Name, Profession

[https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/)

{:.h2_title}
## Data Source
Trained on 10.000 Contact, Location, Name and Profession random replacements

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentificiation.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_large_en_2.5.1_2.4_1595199111307.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DeIdentificationModel.pretrained("deidentify_large","en","clinical/models")\
	.setInputCols("document","token","chunk")\
	.setOutputCol("document")
```

```scala
val model = DeIdentificationModel.pretrained("deidentify_large","en","clinical/models")
	.setInputCols("document","token","chunk")
	.setOutputCol("document")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| name          | deidentify_large       |
| model_class   | DeIdentificationModel  |
| compatibility | 2.5.1                  |
| license       | Licensed               |
| edition       | Healthcare             |
| inputs        | document, token, chunk |
| output        | document               |
| language      | en                     |
| upstream_deps | ner_deid_large         |

