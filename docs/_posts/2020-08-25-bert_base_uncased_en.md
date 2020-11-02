---
layout: model
title: BERT Base Uncased
author: John Snow Labs
name: bert_base_uncased
class: BertEmbeddings
language: en
repository: public/models
date: 2020-08-25
tags: [embeddings]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre,training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_en_2.2.0_2.4_1598340514223.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = BertEmbeddings.pretrained("bert_base_uncased","en","public/models")\
	.setInputCols("document","sentence","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = BertEmbeddings.pretrained("bert_base_uncased","en","public/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|---------------------------|
| Model Name     | bert_base_uncased         |
| Model Class    | BertEmbeddings            |
| Dimension      | 2.4                       |
| Compatibility  | 2.2.0                     |
| License        | open source               |
| Edition        | public                    |
| Inputs         | document, sentence, token |
| Output         | word_embeddings           |
| Language       | en                        |
| dimension      | 768                       |
| Case Sensitive | False                     |
| Dependencies   |                           |




{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1](https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1)  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddings.scala) link to get more information

