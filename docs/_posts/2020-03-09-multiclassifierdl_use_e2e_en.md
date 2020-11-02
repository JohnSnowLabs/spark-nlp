---
layout: model
title: 
author: John Snow Labs
name: multiclassifierdl_use_e2e
class: MultiClassifierDLModel
language: en
repository: public/models
date: 2020-03-09
tags: [classifier]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multiclassifierdl_use_e2e_en_2.6.0_2.4_1599146072149.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = MultiClassifierDLModel.pretrained("multiclassifierdl_use_e2e","en","public/models")\
	.setInputCols("sentence_embeddings","label")\
	.setOutputCol("category")
```

```scala
val model = MultiClassifierDLModel.pretrained("multiclassifierdl_use_e2e","en","public/models")
	.setInputCols("sentence_embeddings","label")
	.setOutputCol("category")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------|
| Model Name     | multiclassifierdl_use_e2e  |
| Model Class    | MultiClassifierDLModel     |
| Dimension      | 2.4                        |
| Compatibility  | 2.6.0                      |
| License        | open source                |
| Edition        | public                     |
| Inputs         | sentence_embeddings, label |
| Output         | category                   |
| Language       | en                         |
| Case Sensitive | True                       |
| Dependencies   | with tfhub_use             |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLModel.scala) link to get more information

