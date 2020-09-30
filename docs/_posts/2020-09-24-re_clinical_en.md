---
layout: model
title: Relation Extraction Model Clinical
author: John Snow Labs
name: re_clinical
class: RelationExtractionModel
language: en
repository: clinical/models
date: 2020-09-24
tags: [clinical,relation,extraction,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Relation Extraction model based on syntactic features using deep learning  
Models the set of clinical relations defined in the 2010 i2b2 relation challenge.

{:.h2_title}
## Prediction Labels
TrIP (improved), TrWP (worsened), TrCP (caused problem), TrAP (administered), TrNAP (avoided), TeRP (revealed problem), TeCP (investigate problem), PIP (problems related)

[https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL/){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_clinical_en_2.5.5_2.4_1600987935304.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = RelationExtractionModel.pretrained("re_clinical","en","clinical/models") \
	.setInputCols("word_embeddings","chunk","pos","dependency") \
	.setOutputCol("category")
```

```scala
val model = RelationExtractionModel.pretrained("re_clinical","en","clinical/models")
	.setInputCols("word_embeddings","chunk","pos","dependency")
	.setOutputCol("category")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------------------|
| Model Name     | re_clinical                             |
| Type           | RelationExtractionModel                 |
| Compatibility  | 2.5.5                                   |
| License        | Licensed                                |
| Edition        | Healthcare                              |
| Inputs         | word_embeddings, chunk, pos, dependency |
| Output         | category                                |
| Language       | en                                      |
| Case Sensitive | False                                   |
| Dependencies   | embeddings_clinical                     |

