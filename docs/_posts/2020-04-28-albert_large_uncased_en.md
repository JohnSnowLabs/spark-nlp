---
layout: model
title: ALBERT Large Uncase
author: John Snow Labs
name: albert_large_uncased
class: AlbertEmbeddings
language: en
repository: public/models
date: 2020-04-28
tags: [embeddings]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm. ALBERT uses parameter,reduction techniques that allow for large,scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation. The details are described in the paper "[ALBERT: A Lite BERT for Self,supervised Learning of Language Representations.](https://arxiv.org/abs/1909.11942)"



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_large_uncased_en_2.5.0_2.4_1588073397355.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = AlbertEmbeddings.pretrained("albert_large_uncased","en","public/models")\
	.setInputCols("document","sentence","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = AlbertEmbeddings.pretrained("albert_large_uncased","en","public/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|---------------------------|
| Model Name     | albert_large_uncased      |
| Model Class    | AlbertEmbeddings          |
| Dimension      | 2.4                       |
| Compatibility  | 2.5.0                     |
| License        | open source               |
| Edition        | public                    |
| Inputs         | document, sentence, token |
| Output         | word_embeddings           |
| Language       | en                        |
| dimension      | 1024                      |
| Case Sensitive | False                     |
| Dependencies   |                           |




{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/albert_large/3](https://tfhub.dev/google/albert_large/3)  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/AlbertEmbeddings.scala) link to get more information

