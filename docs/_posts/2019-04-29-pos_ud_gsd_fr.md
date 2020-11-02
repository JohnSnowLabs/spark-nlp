---
layout: model
title: 
author: John Snow Labs
name: pos_ud_gsd
class: PerceptronModel
language: fr
repository: public/models
date: 2019-04-29
tags: [pos]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_fr_2.0.2_2.4_1556531457346.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = PerceptronModel.pretrained("pos_ud_gsd","fr","public/models")\
	.setInputCols("token","sentence")\
	.setOutputCol("pos")
```

```scala
val model = PerceptronModel.pretrained("pos_ud_gsd","fr","public/models")
	.setInputCols("token","sentence")
	.setOutputCol("pos")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-----------------|
| Model Name     | pos_ud_gsd      |
| Model Class    | PerceptronModel |
| Dimension      | 2.4             |
| Compatibility  | 2.0.2           |
| License        | open source     |
| Edition        | public          |
| Inputs         | token, sentence |
| Output         | pos             |
| Language       | fr              |
| Case Sensitive | True            |
| Dependencies   | POS UD          |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/PerceptronModel.scala) link to get more information

