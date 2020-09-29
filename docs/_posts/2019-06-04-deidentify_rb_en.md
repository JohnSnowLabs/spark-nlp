---
layout: model
title: Deidentify RB
author: John Snow Labs
name: deidentify_rb
class: DeIdentificationModel
language: en
repository: clinical/models
date: 2019-06-04
tags: [clinical,deidentification,regex,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Anonymization and DeIdentification model based on outputs from DeId NERs and Replacement Dictionaries


{:.h2_title}
## Prediction Domain
Personal Information in order to deidentify

{:.h2_title}
## Data Source
Rule based DeIdentifier based on `ner_deid`


{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_en_2.0.2_2.4_1559672122511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DeIdentificationModel.pretrained("deidentify_rb","en","clinical/models")
	.setInputCols("document","token","chunk")
	.setOutputCol("document")
```

```scala
val model = DeIdentificationModel.pretrained("deidentify_rb","en","clinical/models")
	.setInputCols("document","token","chunk")
	.setOutputCol("document")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| name          | deidentify_rb          |
| model_class   | DeIdentificationModel  |
| compatibility | 2.0.2                  |
| license       | Licensed               |
| edition       | Healthcare             |
| inputs        | document, token, chunk |
| output        | document               |
| language      | en                     |
| upstream_deps | ner_deid               |

