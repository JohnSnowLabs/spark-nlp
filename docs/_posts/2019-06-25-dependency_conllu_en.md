---
layout: model
title: 
author: John Snow Labs
name: dependency_conllu
class: DependencyParser
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
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_2.0.2_2.4_1561435004077.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DependencyParser.pretrained("dependency_conllu","en","public/models")\
	.setInputCols("")\
	.setOutputCol("")
```

```scala
val model = DependencyParser.pretrained("dependency_conllu","en","public/models")
	.setInputCols("")
	.setOutputCol("")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-------------------|
| Model Name     | dependency_conllu |
| Model Class    | DependencyParser  |
| Dimension      | 2.4               |
| Compatibility  | 2.0.2             |
| License        | open source       |
| Edition        | public            |
| Inputs         |                   |
| Output         |                   |
| Language       | en                |
| Case Sensitive | True              |
| Dependencies   | Dependency        |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

