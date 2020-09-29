---
layout: model
title: Ner DL Model Bionlp
author: John Snow Labs
name: ner_bionlp
class: NerDLModel
language: en
repository: clinical/models
date: 2020-01-28
tags: [clinical,ner,dl,cancer,genetics,bionlp,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Pretrained named entity recognition deep learning model for biology and genetics terms.

{:.h2_title}
## Prediction Domain
Amino_acid,Anatomical_system,Cancer,Cell,Cellular_component,Developing_anatomical_structure,Gene_or_gene_product,Immaterial_anatomical_entity,Organ,Organism,Organism_subdivision,Organism_substance,Pathological_formation,Simple_chemical,Tissue,tissue_structure

{:.h2_title}
## Data Source
Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013 with `embeddings_clinical`
http://2013.bionlp-st.org/tasks/cancer-genetics

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_bionlp","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_bionlp","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| name           | ner_bionlp                       |
| model_class    | NerDLModel                       |
| compatibility  | 2.4.0                            |
| license        | Licensed                         |
| edition        | Healthcare                       |
| inputs         | sentence, token, word_embeddings |
| output         | ner                              |
| language       | en                               |
| case_sensitive | False                            |
| upstream_deps  | embeddings_clinical              |

