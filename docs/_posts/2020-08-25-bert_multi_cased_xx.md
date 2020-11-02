---
layout: model
title: 
author: John Snow Labs
name: bert_multi_cased
class: BertEmbeddings
language: xx
repository: public/models
date: 2020-08-25
tags: [embeddings]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_multi_cased_xx_2.0.3_2.4_1598341875191.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = BertEmbeddings.pretrained("bert_multi_cased","xx","public/models")\
	.setInputCols("document","sentence","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = BertEmbeddings.pretrained("bert_multi_cased","xx","public/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|---------------------------|
| Model Name     | bert_multi_cased          |
| Model Class    | BertEmbeddings            |
| Dimension      | 2.4                       |
| Compatibility  | 2.0.3                     |
| License        | open source               |
| Edition        | public                    |
| Inputs         | document, sentence, token |
| Output         | word_embeddings           |
| Language       | xx                        |
| Case Sensitive | True                      |
| Dependencies   |                           |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddings.scala) link to get more information

