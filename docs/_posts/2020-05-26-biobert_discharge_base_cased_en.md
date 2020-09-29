---
layout: model
title: Biobert Discharge Base Cased
author: John Snow Labs
name: biobert_discharge_base_cased
class: BertEmbeddings
language: en
repository: clinical/models
date: 2020-05-26
tags: [clinical,bert,embeddings,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture
Contextual embeddings representation using biobert_discharge_base_cased

{:.h2_title}
## Prediction Domain
Contextual feature vectors based on biobert_discharge_base_cased

{:.h2_title}
## Data Source
Trained on PubMed + MIMIC III corpora
https://github.com/naver/biobert-pretrained

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobert_discharge_base_cased_en_2.5.0_2.4_1590490193605.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = BertEmbeddings.pretrained("biobert_discharge_base_cased","en","clinical/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```

```scala
val model = BertEmbeddings.pretrained("biobert_discharge_base_cased","en","clinical/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|------------------------------|
| name           | biobert_discharge_base_cased |
| model_class    | BertEmbeddings               |
| compatibility  | 2.5.0                        |
| license        | Licensed                     |
| edition        | Healthcare                   |
| inputs         | document, sentence, token    |
| output         | word_embeddings              |
| language       | en                           |
| dimension      | 768.0                        |
| case_sensitive | True                         |

