---
layout: model
title: 
author: John Snow Labs
name: dependency_typed_conllu
class: TypedDependencyParser
language: en
repository: public/models
date: 2019-06-25
tags: []
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_typed_conllu_en_2.0.2_2.4_1561473259215.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = TypedDependencyParser.pretrained("dependency_typed_conllu","en","public/models")\
	.setInputCols("")\
	.setOutputCol("")
```

```scala
val model = TypedDependencyParser.pretrained("dependency_typed_conllu","en","public/models")
	.setInputCols("")
	.setOutputCol("")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-------------------------|
| Model Name     | dependency_typed_conllu |
| Model Class    | TypedDependencyParser   |
| Dimension      | 2.4                     |
| Compatibility  | 2.0.2                   |
| License        | open source             |
| Edition        | public                  |
| Inputs         |                         |
| Output         |                         |
| Language       | en                      |
| Case Sensitive | True                    |
| Dependencies   | Dependency              |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

