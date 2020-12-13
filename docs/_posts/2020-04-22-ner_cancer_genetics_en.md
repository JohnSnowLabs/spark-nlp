---
layout: model
title: Cancer Genetics Entity Extracter
author: John Snow Labs
name: ner_cancer_genetics
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-22
tags: [clinical,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Pretrained named entity recognition deep learning model for biology and genetics terms.

## Predicted Entities 
DNA, RNA, cell_line, cell_type, protein

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_cancer_genetics_en_2.4.2_2.4_1587567870408.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_cancer_genetics","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_cancer_genetics","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>


{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------------|
| Name:          | ner_cancer_genetics              |
| Type:   | NerDLModel                       |
| Compatibility: | 2.4.2                            |
| License:       | Licensed                         |
| Edition:       | Official                       |
|Input labels:        | sentence, token, word_embeddings |
|Output labels:       | ner                              |
| Language:      | en                               |
| Dependencies: | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013 with `embeddings_clinical`
http://2013.bionlp-st.org/tasks/cancer-genetics