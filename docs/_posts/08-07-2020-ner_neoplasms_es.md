---
layout: model
title: Neoplasms NER
author: John Snow Labs
name: ner_neoplasms
class: NerDLModel
language: es
repository: clinical/models
date: 08/07/2020
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Neoplasms NER is a Named Entity Recognition model that annotates text to find references to tumors. The only entity it annotates is MalignantNeoplasm. Neoplasms NER is trained with the 'embeddings_scielowiki_300d' word embeddings model, so be sure to use the same embeddings in the pipeline.

 {:.h2_title}
## Predicted Entities
MORFOLOGIA_NEOPLASIA 



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
Named Entity Recognition model for Neoplasic Morphology

