---
layout: model
title: 
author: John Snow Labs
name: classifierdl_use_spam
class: ClassifierDLModel
language: en
repository: public/models
date: 2020-03-07
tags: [classifier]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_spam_en_2.5.3_2.4_1593783318934.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ClassifierDLModel.pretrained("classifierdl_use_spam","en","public/models")\
	.setInputCols("sentence_embeddings","label")\
	.setOutputCol("category")
```

```scala
val model = ClassifierDLModel.pretrained("classifierdl_use_spam","en","public/models")
	.setInputCols("sentence_embeddings","label")
	.setOutputCol("category")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------|
| Model Name     | classifierdl_use_spam      |
| Model Class    | ClassifierDLModel          |
| Dimension      | 2.4                        |
| Compatibility  | 2.5.3                      |
| License        | open source                |
| Edition        | public                     |
| Inputs         | sentence_embeddings, label |
| Output         | category                   |
| Language       | en                         |
| Case Sensitive | True                       |
| Dependencies   | with tfhub_use             |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLModel.scala) link to get more information

