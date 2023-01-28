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
edition: Healthcare NLP 2.4.2
spark_version: 2.4
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
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/6.Clinical_Context_Spell_Checker.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.4.2_2.4_1587146727460.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.4.2_2.4_1587146727460.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

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


{:.nlu-block}
```python
import nlu
nlu.load("en.spell.clinical").predict("""Put your text here.""")
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
