---
layout: model
title: 
author: John Snow Labs
name: ner_posology_healthcare
class: NerDLModel
language: en
repository: clinical/models
date: 23/09/2020
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Pretrained named entity recognition deep learning model for posology, this NER is trained with the 'embeddings_healthcare' word embeddings model, so be sure to use the same embeddings in the pipeline

 {:.h2_title}
## Predicted Entities
Dosage,Drug,Duration,Form,Frequency,Route,Strength 



## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

```

```scala

```
</div>



{:.model-param}
## Model Information
{:.table-model}





{:.h2_title}
## Data Source
Trained on the 2018 i2b2 dataset and FDA Drug datasets with `embeddings_healthcare_100d`.

