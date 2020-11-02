---
layout: model
title: 
author: John Snow Labs
name: sentimentdl_use_imdb
class: SentimentDLModel
language: en
repository: public/models
date: 2020-08-06
tags: [sentiment]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_use_imdb_en_2.5.0_2.4_1591608094321.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentimentDLModel.pretrained("sentimentdl_use_imdb","en","public/models")\
	.setInputCols("sentence","label","sentence_embeddings")\
	.setOutputCol("category")
```

```scala
val model = SentimentDLModel.pretrained("sentimentdl_use_imdb","en","public/models")
	.setInputCols("sentence","label","sentence_embeddings")
	.setOutputCol("category")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|--------------------------------------|
| Model Name     | sentimentdl_use_imdb                 |
| Model Class    | SentimentDLModel                     |
| Dimension      | 2.4                                  |
| Compatibility  | 2.5.0                                |
| License        | open source                          |
| Edition        | public                               |
| Inputs         | sentence, label, sentence_embeddings |
| Output         | category                             |
| Language       | en                                   |
| Case Sensitive | True                                 |
| Dependencies   | with tfhub_use                       |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLModel.scala) link to get more information

