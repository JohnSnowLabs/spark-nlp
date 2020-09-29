---
layout: model
title: NER DL Model Legal
author: John Snow Labs
name: ner_legal
class: NerDLModel
language: de
repository: clinical/models
date: 2020-09-07
tags: [legal,ner,de]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  


{:.h2_title}
## Prediction Domain
AN,EUN,GRT,GS,INN,LD,LDS,LIT,MRK,ORG,PER,RR,RS,ST,STR,UN,VO,VS,VT



{:.h2_title}
## Data Source


{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/15.German_Legal_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_legal_de_2.5.5_2.4_1599471454959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_legal","de","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_legal","de","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| name          | ner_legal                        |
| model_class   | NerDLModel                       |
| compatibility | 2.5.5                            |
| license       | Licensed                         |
| edition       | Legal                            |
| inputs        | sentence, token, word_embeddings |
| output        | ner                              |
| language      | de                               |
| upstream_deps | embeddings_clinical              |

