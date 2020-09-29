---
layout: model
title: Deidentify RB No Regex
author: John Snow Labs
name: deidentify_rb_no_regex
class: DeIdentificationModel
language: en
repository: clinical/models
date: 2020-05-19
tags: [clinical, deidentify, en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Anonymization and DeIdentification model based on outputs from DeId NERs and Replacement Dictionaries.

## Predicted Entities 
Personal Information in order to deidentify.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_no_regex_en_2.5.0_2.4_1589924063833.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DeIdentificationModel.pretrained("deidentify_rb_no_regex","en","clinical/models")
	.setInputCols("document","token","chunk")
	.setOutputCol("document")
```

```scala
val model = DeIdentificationModel.pretrained("deidentify_rb_no_regex","en","clinical/models")
	.setInputCols("document","token","chunk")
	.setOutputCol("document")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| Name:          | deidentify_rb_no_regex |
| Type:   | DeIdentificationModel  |
| Compatibility: | 2.4.5                  |
| License:       | Licensed               |
| Edition:       | Official             |
|Input labels:        | [document, token, chunk] |
|Output labels:       | [document]               |
| Language:      | en                     |
| Dependencies: | ner_deid               |

{:.h2_title}
## Data Source
Rule based DeIdentifier based on `ner_deid`.