---
layout: model
title: Ner DL Model Clinical (Large)
author: John Snow Labs
name: ner_clinical_large
class: NerDLModel
language: en
repository: clinical/models
date: 21/05/2020
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Clinical NER (Large) is a Named Entity Recognition model that annotates text to find references to clinical events. The entities it annotates are Problem, Treatment, and Test. Clinical NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

 {:.h2_title}
## Predicted Entities
Problem, Test, Treatment 



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
Trained on i2b2 augmented data with `clinical_embeddings`

