---
layout: model
title: 
author: John Snow Labs
name: spellcheck_norvig
class: NorvigSweetingModel
language: en
repository: public/models
date: 2019-07-13
tags: []
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_norvig_en_2.0.2_2.4_1563017660080.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NorvigSweetingModel.pretrained("spellcheck_norvig","en","public/models")\
	.setInputCols("token")\
	.setOutputCol("token")
```

```scala
val model = NorvigSweetingModel.pretrained("spellcheck_norvig","en","public/models")
	.setInputCols("token")
	.setOutputCol("token")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|---------------------|
| Model Name     | spellcheck_norvig   |
| Model Class    | NorvigSweetingModel |
| Dimension      | 2.4                 |
| Compatibility  | 2.0.2               |
| License        | open source         |
| Edition        | public              |
| Inputs         | token               |
| Output         | token               |
| Language       | en                  |
| Case Sensitive | True                |
| Dependencies   | Spell Checker       |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig) link to get more information

