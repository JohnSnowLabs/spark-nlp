---
layout: model
title: Clinical Ner Assertion
author: John Snow Labs
name: clinical_ner_assertion
class: PipelineModel
language: en
repository: clinical/models
date: 2020-01-31
tags: []
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description



{:.h2_title}
## Data Source



{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_ner_assertion_en_2.4.0_2.4_1580481098096.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PipelineModel.pretrained("clinical_ner_assertion","en","clinical/models")
	.setInputCols("")
	.setOutputCol("")
```

```scala
val model = PipelineModel.pretrained("clinical_ner_assertion","en","clinical/models")
	.setInputCols("")
	.setOutputCol("")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| name          | clinical_ner_assertion |
| model_class   | PipelineModel          |
| compatibility | 2.4.0                  |
| license       | Licensed               |
| edition       | Healthcare             |
| language      | en                     |

