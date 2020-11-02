---
layout: model
title: Assertion with Biobert Embeddings
author: John Snow Labs
name: assertion_dl_biobert
class: AssertionDLModel
language: en
repository: clinical/models
date: 2020-06-10
tags: [clinical,assertion]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Deep learning named entity recognition model for assertions 

 {:.h2_title}
## Predicted Entities
hypothetical, present, absent, possible, conditional, associated_with_someone_else 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button><br/>[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertiondl_biobert_en_2.6.1_2.4_1601928965625.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = AssertionDLModel.pretrained("assertion_dl_biobert","en","clinical/models")\
	.setInputCols("document","chunk","word_embeddings")\
	.setOutputCol("assertion")
```

```scala
val model = AssertionDLModel.pretrained("assertion_dl_biobert","en","clinical/models")
	.setInputCols("document","chunk","word_embeddings")
	.setOutputCol("assertion")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|----------------------------------|
| Model Name     | assertion_dl_biobert             |
| Model Class    | AssertionDLModel                 |
| Dimension      | 2.4                              |
| Compatibility  | 2.6.2                            |
| License        | Licensed                         |
| Edition        | Healthcare                       |
| Inputs         | document, chunk, word_embeddings |
| Output         | assertion                        |
| Language       | en                               |
| Case Sensitive | False                            |
| Dependencies   | biobert_pubmed_base_cased        |




{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with `biobert_pubmed_base_cased`  
Visit [this](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) link to get more information

