---
layout: model
title: XLNet Large
author: John Snow Labs
name: xlnet_large_cased
class: XlnetEmbeddings
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
XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer,XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state,of,the,art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking. The details are described in the paper "[‚ÄãXLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)"



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_large_cased_en_2.5.0_2.4_1588074397954.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = XlnetEmbeddings.pretrained("xlnet_large_cased","en","public/models")\
	.setInputCols("document","sentence","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = XlnetEmbeddings.pretrained("xlnet_large_cased","en","public/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|---------------------------|
| Model Name     | xlnet_large_cased         |
| Model Class    | XlnetEmbeddings           |
| Dimension      | 2.4                       |
| Compatibility  | 2.5.0                     |
| License        | open source               |
| Edition        | public                    |
| Inputs         | document, sentence, token |
| Output         | word_embeddings           |
| Language       | en                        |
| dimension      | 1024                      |
| Case Sensitive | True                      |
| Dependencies   |                           |




{:.h2_title}
## Data Source
The model is imported from [https://github.com/zihangdai/xlnet](https://github.com/zihangdai/xlnet)  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/XlnetEmbeddings.scala) link to get more information

