---
layout: model
title: NER Adverse Drug Events
author: John Snow Labs
name: ner_ade_healthcare
class: NerDLModel
language: en
repository: clinical/models
date: 2020-06-10
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Extract adverse drug reaction events and drug entites from text

 {:.h2_title}
## Predicted Entities
ADE, DRUG 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_ade_healthcare_en_2.6.0_2.4_1601450601043.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_ade_healthcare","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_ade_healthcare","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------------|
| Model Name     | ner_ade_healthcare               |
| Model Class    | NerDLModel                       |
| Dimension      | 2.4                              |
| Compatibility  | 2.6.2                            |
| License        | Licensed                         |
| Edition        | Healthcare                       |
| Inputs         | sentence, token, word_embeddings |
| Output         | ner                              |
| Language       | en                               |
| Case Sensitive | True                             |
| Dependencies   | embeddings_healthcare_100d       |




{:.h2_title}
## Data Source
Trained on DRUG-AE, 2018 i2b2, CADEC, and twitter ADE dataset  
Visit [this](FILLUP) link to get more information

