---
layout: model
title: 
author: John Snow Labs
name: ner_dl_sentence
class: DeepSentenceDetector
language: en
repository: public/models
date: 2020-01-28
tags: [sentence_detector]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 




{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_dl_sentence_en_2.0.2_2.4_1580252313303.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DeepSentenceDetector.pretrained("ner_dl_sentence","en","public/models")\
	.setInputCols("document")\
	.setOutputCol("sentence")
```

```scala
val model = DeepSentenceDetector.pretrained("ner_dl_sentence","en","public/models")
	.setInputCols("document")
	.setOutputCol("sentence")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------|
| Model Name     | ner_dl_sentence      |
| Model Class    | DeepSentenceDetector |
| Dimension      | 2.4                  |
| Compatibility  | 2.0.2                |
| License        | open source          |
| Edition        | public               |
| Inputs         | document             |
| Output         | sentence             |
| Language       | en                   |
| Case Sensitive | True                 |
| Dependencies   |                      |




{:.h2_title}
## Data Source
  
Visit [this](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/deep/DeepSentenceDetector.scala) link to get more information

