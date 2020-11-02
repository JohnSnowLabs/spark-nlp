---
layout: model
title: Ner DL Model Anatomy
author: John Snow Labs
name: ner_anatomy
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Pretrained named entity recognition deep learning model for anatomy terms.

 {:.h2_title}
## Predicted Entities
Anatomical_system,Cell,Cellular_component,Developing_anatomical_structure,Immaterial_anatomical_entity,Organ,Organism_subdivision,Organism_substance,Pathological_formation,Tissue,tissue_structure 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_en_2.4.2_2.4_1587513307751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_anatomy","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_anatomy","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------------|
| Model Name     | ner_anatomy                      |
| Model Class    | NerDLModel                       |
| Dimension      | 2.4                              |
| Compatibility  | 2.4.2                            |
| License        | Licensed                         |
| Edition        | Healthcare                       |
| Inputs         | sentence, token, word_embeddings |
| Output         | ner                              |
| Language       | en                               |
| Case Sensitive | False                            |
| Dependencies   | embeddings_clinical              |




{:.h2_title}
## Data Source
Trained on the Anatomical Entity Mention (AnEM) corpus with `embeddings_clinical`  
Visit [this](http://www.nactem.ac.uk/anatomy/) link to get more information

