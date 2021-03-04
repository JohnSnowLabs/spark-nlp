---
layout: model
title: Contextual SpellChecker Clinical
author: John Snow Labs
name: spellcheck_clinical
class: ContextSpellCheckerModel
language: en
repository: clinical/models
date: 2020-04-17
task: Spell Check
edition: Spark NLP for Healthcare 2.4.2
tags: [clinical,licensed,en]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Implements Noisy Channel Model Spell Algorithm. Correction candidates are extracted combining context information and word information

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/6.Clinical_Context_Spell_Checker.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.4.2_2.4_1587146727460.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ContextSpellCheckerModel.pretrained("spellcheck_clinical","en","clinical/models")
	.setInputCols("token")
	.setOutputCol("spell")
```

```scala
val model = ContextSpellCheckerModel.pretrained("spellcheck_clinical","en","clinical/models")
	.setInputCols("token")
	.setOutputCol("spell")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|--------------------------|
| Name:          | spellcheck_clinical      |
| Type:   | ContextSpellCheckerModel |
| Compatibility: | 2.4.2                    |
| License:       | Licensed                 |
| Edition:       | Official               |
|Input labels:        | [token]                    |
|Output labels:       | [spell]                    |
| Language:      | en                       |
| Dependencies: | embeddings_clinical      |

{:.h2_title}
## Data Source
Trained with PubMed and i2b2 datasets.
