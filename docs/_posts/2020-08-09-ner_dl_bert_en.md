---
layout: model
title: 
author: John Snow Labs
name: ner_dl_bert
class: NerDLModel
language: en
repository: public/models
date: 2020-08-09
tags: [ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_bert_en_2.0.2_2.4_1599550979101.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_dl_bert","en","public/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_dl_bert","en","public/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------------|
| Model Name     | ner_dl_bert                      |
| Model Class    | NerDLModel                       |
| Dimension      | 2.4                              |
| Compatibility  | 2.0.2                            |
| License        | open source                      |
| Edition        | public                           |
| Inputs         | sentence, token, word_embeddings |
| Output         | ner                              |
| Language       | en                               |
| Case Sensitive | True                             |
| Dependencies   | NER with BERT                    |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLModel.scala) link to get more information

