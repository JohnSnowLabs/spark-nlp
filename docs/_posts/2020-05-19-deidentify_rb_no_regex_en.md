---
layout: model
title: Deidentify RB No Regex
author: John Snow Labs
name: deidentify_rb_no_regex
class: DeIdentificationModel
language: en
repository: clinical/models
date: 2020-05-19
tags: [clinical,deidentification]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 


 {:.h2_title}
## Predicted Entities
Personal Information in order to deidentify 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_no_regex_en_2.5.0_2.4_1589924063833.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DeIdentificationModel.pretrained("deidentify_rb_no_regex","en","clinical/models")\
	.setInputCols("document","token","chunk")\
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
|----------------|------------------------|
| Model Name     | deidentify_rb_no_regex |
| Model Class    | DeIdentificationModel  |
| Dimension      | 2.4                    |
| Compatibility  | 2.4.5                  |
| License        | Licensed               |
| Edition        | Healthcare             |
| Inputs         | document, token, chunk |
| Output         | document               |
| Language       | en                     |
| Case Sensitive | True                   |
| Dependencies   | ner_deid               |




{:.h2_title}
## Data Source
Rule based DeIdentifier based on `ner_deid`  
Visit [this]() link to get more information

