---
layout: model
title: POS Tagger Clinical
author: John Snow Labs
name: pos_clinical
class: PerceptronModel
language: en
repository: clinical/models
date: 2019-04-30
tags: [clinical, licensed, pos,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Sets a Part-Of-Speech tag to each word within a sentence.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/pos_clinical_en_2.0.2_2.4_1556660550177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PerceptronModel.pretrained("pos_clinical","en","clinical/models")\
	.setInputCols("token","sentence")\
	.setOutputCol("pos")
```

```scala
val model = PerceptronModel.pretrained("pos_clinical","en","clinical/models")
	.setInputCols("token","sentence")
	.setOutputCol("pos")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:          | pos_clinical        |
| Type:   | PerceptronModel     |
| Compatibility: | Spark NLP 2.0.2+               |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [token, sentence]     |
|Output labels:       | [pos]                 |
| Language:      | en                  |
| Dependencies: | embeddings_clinical |

{:.h2_title}
## Data Source
Trained with MedPost dataset.