---
layout: model
title: 
author: John Snow Labs
name: sentiment_vivekn
class: ViveknSentimentModel
language: en
repository: public/models
date: 2019-04-30
tags: [sentiment]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_vivekn_en_2.0.2_2.4_1556663184035.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ViveknSentimentModel.pretrained("sentiment_vivekn","en","public/models")\
	.setInputCols("sentence","sentiment_label","token")\
	.setOutputCol("sentiment")
```

```scala
val model = ViveknSentimentModel.pretrained("sentiment_vivekn","en","public/models")
	.setInputCols("sentence","sentiment_label","token")
	.setOutputCol("sentiment")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------------|
| Model Name     | sentiment_vivekn                 |
| Model Class    | ViveknSentimentModel             |
| Dimension      | 2.4                              |
| Compatibility  | 2.0.2                            |
| License        | open source                      |
| Edition        | public                           |
| Inputs         | sentence, sentiment_label, token |
| Output         | sentiment                        |
| Language       | en                               |
| Case Sensitive | True                             |
| Dependencies   | Sentiment                        |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentModel.scala) link to get more information

