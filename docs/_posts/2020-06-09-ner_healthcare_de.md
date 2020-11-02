---
layout: model
title: w2v_cc_300d
author: John Snow Labs
name: ner_healthcare
class: NerDLModel
language: de
repository: clinical/models
date: 2020-06-09
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 


 {:.h2_title}
## Predicted Entities
BIOLOGICAL_CHEMISTRY,BIOLOGICAL_PARAMETER,BODY_FLUID,BODY_PART,DEGREE,DIAGLAB_PROCEDURE,DOSING,LOCAL_SPECIFICATION,MEASUREMENT,MEDICAL_CONDITION,MEDICAL_DEVICE,MEDICAL_SPECIFICATION,MEDICATION,PERSON,PROCESS,STATE_OF_HEALTH,TIME_INFORMATION,TISSUE,TREATMENT 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_2.5.5_2.4_1599433028253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_healthcare","de","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_healthcare","de","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------------|
| Model Name     | ner_healthcare                   |
| Model Class    | NerDLModel                       |
| Dimension      | 2.4                              |
| Compatibility  | 2.5.5                            |
| License        | Licensed                         |
| Edition        | Healthcare                       |
| Inputs         | sentence, token, word_embeddings |
| Output         | ner                              |
| Language       | de                               |
| Case Sensitive | True                             |
| Dependencies   | FILLUP                           |




{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with `w2v_cc_300d`  
Visit [this]() link to get more information

