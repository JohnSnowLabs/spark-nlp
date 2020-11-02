---
layout: model
title: 
author: John Snow Labs
name: spellcheck_sd
class: SymmetricDeleteModel
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
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_sd_en_2.0.2_2.4_1563019290368.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SymmetricDeleteModel.pretrained("spellcheck_sd","en","public/models")\
	.setInputCols("token")\
	.setOutputCol("spell")
```

```scala
val model = SymmetricDeleteModel.pretrained("spellcheck_sd","en","public/models")
	.setInputCols("token")
	.setOutputCol("spell")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------|
| Model Name     | spellcheck_sd        |
| Model Class    | SymmetricDeleteModel |
| Dimension      | 2.4                  |
| Compatibility  | 2.0.2                |
| License        | open source          |
| Edition        | public               |
| Inputs         | token                |
| Output         | spell                |
| Language       | en                   |
| Case Sensitive | True                 |
| Dependencies   | Spell Checker        |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric) link to get more information

