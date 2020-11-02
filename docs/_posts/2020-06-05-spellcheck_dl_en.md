---
layout: model
title: 
author: John Snow Labs
name: spellcheck_dl
class: ContextSpellCheckerModel
language: en
repository: public/models
date: 2020-06-05
tags: [spell_checker]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_dl_en_2.4.2_2.4_1588756259065.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ContextSpellCheckerModel.pretrained("spellcheck_dl","en","public/models")\
	.setInputCols("token")\
	.setOutputCol("spell")
```

```scala
val model = ContextSpellCheckerModel.pretrained("spellcheck_dl","en","public/models")
	.setInputCols("token")
	.setOutputCol("spell")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|--------------------------|
| Model Name     | spellcheck_dl            |
| Model Class    | ContextSpellCheckerModel |
| Dimension      | 2.4                      |
| Compatibility  | 2.4.2                    |
| License        | open source              |
| Edition        | public                   |
| Inputs         | token                    |
| Output         | spell                    |
| Language       | en                       |
| Case Sensitive | True                     |
| Dependencies   | Spell Checker            |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/context) link to get more information

