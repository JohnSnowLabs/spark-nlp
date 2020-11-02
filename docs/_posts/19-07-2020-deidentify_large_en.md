---
layout: model
title: Deidentify Large
author: John Snow Labs
name: deidentify_large
class: DeIdentificationModel
language: en
repository: clinical/models
date: 19/07/2020
tags: [clinical,deidentification]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Deidentify (Large) is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing "2020,06,04" with "<DATE>"). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

 {:.h2_title}
## Predicted Entities
Contact, Location, Name, Profession 



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
Trained on 10.000 Contact, Location, Name and Profession random replacements

