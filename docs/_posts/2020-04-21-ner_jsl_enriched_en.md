---
layout: model
title: Ner DL Model Enriched
author: John Snow Labs
name: ner_jsl_enriched
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,ner,generic,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.  
Pretrained named entity recognition deep learning model for clinical terminology.

{:.h2_title}
## Prediction Domain
Age,Allergenic_substance,Blood_Pressure,Causative_Agents_(Virus_and_Bacteria),Diagnosis,Dosage,Drug_Name,Frequency,Gender,Lab_Name,Lab_Result,Maybe,Modifier,Name,Negation,O2_Saturation,Procedure,Procedure_Name,Pulse_Rate,Respiratory_Rate,Route,Section_Name,Substance_Name,Symptom_Name,Temperature,Weight

[https://www.johnsnowlabs.com/data/](https://www.johnsnowlabs.com/data/)

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs

{:.btn-box}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_en_2.4.2_2.4_1587513303751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_jsl_enriched","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_jsl_enriched","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| name           | ner_jsl_enriched                 |
| model_class    | NerDLModel                       |
| compatibility  | 2.4.2                            |
| license        | Licensed                         |
| edition        | Healthcare                       |
| inputs         | sentence, token, word_embeddings |
| output         | ner                              |
| language       | en                               |
| case_sensitive | False                            |
| upstream_deps  | embeddings_clinical              |

