---
layout: model
title: Ner DL Model Phenotype / Gene
author: John Snow Labs
name: ner_human_phenotype_gene_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-08-27
tags: [clinical,ner]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 


 {:.h2_title}
## Predicted Entities
GENE,HP 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL/){:.button.button-orange}<br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_2.5.5_2.4_1598558253840.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_human_phenotype_gene_clinical","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
```

```scala
val model = NerDLModel.pretrained("ner_human_phenotype_gene_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|-----------------------------------|
| Model Name     | ner_human_phenotype_gene_clinical |
| Model Class    | NerDLModel                        |
| Dimension      | 2.4                               |
| Compatibility  | 2.5.5                             |
| License        | Licensed                          |
| Edition        | Healthcare                        |
| Inputs         | sentence, token, word_embeddings  |
| Output         | ner                               |
| Language       | en                                |
| Case Sensitive | True                              |
| Dependencies   | embeddings_clinical               |




{:.h2_title}
## Data Source
  
Visit [this]() link to get more information

