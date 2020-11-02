---
layout: model
title: 
author: John Snow Labs
name: lemma_antbnc
class: LemmatizerModel
language: en
repository: public/models
date: 2019-04-28
tags: [lemmatizer]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_antbnc_en_2.0.2_2.4_1556480454569.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = LemmatizerModel.pretrained("lemma_antbnc","en","public/models")\
	.setInputCols("token")\
	.setOutputCol("lemma")
```

```scala
val model = LemmatizerModel.pretrained("lemma_antbnc","en","public/models")
	.setInputCols("token")
	.setOutputCol("lemma")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-----------------|
| Model Name     | lemma_antbnc    |
| Model Class    | LemmatizerModel |
| Dimension      | 2.4             |
| Compatibility  | 2.0.2           |
| License        | open source     |
| Edition        | public          |
| Inputs         | token           |
| Output         | lemma           |
| Language       | en              |
| Case Sensitive | True            |
| Dependencies   | Lemmatizer      |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/LemmatizerModel.scala) link to get more information

