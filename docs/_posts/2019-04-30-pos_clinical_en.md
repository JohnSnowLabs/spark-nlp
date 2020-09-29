---
layout: model
title: POS Tagger Clinical
author: John Snow Labs
name: pos_clinical
class: PerceptronModel
language: en
repository: clinical/models
date: 2019-04-30
tags: [clinical,pos,medpost,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Sets a Part-Of-Speech tag to each word within a sentence.  




{:.h2_title}
## Data Source
Trained with MedPost dataset

{:.btn-box}
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
| name          | pos_clinical        |
| model_class   | PerceptronModel     |
| compatibility | 2.0.2               |
| license       | Licensed            |
| edition       | Healthcare          |
| inputs        | token, sentence     |
| output        | pos                 |
| language      | en                  |
| upstream_deps | embeddings_clinical |

