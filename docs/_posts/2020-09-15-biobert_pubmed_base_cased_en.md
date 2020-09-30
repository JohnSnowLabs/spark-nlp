---
layout: model
title: Biobert Pubmed Base Cased
author: John Snow Labs
name: biobert_pubmed_base_cased
class: BertEmbeddings
language: en
repository: clinical/models
date: 2020-09-15
tags: [clinical,bert,embeddings,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture  
Contextual embeddings representation using biobert_pubmed_base_cased

{:.h2_title}
## Prediction Labels
Contextual feature vectors based on biobert_pubmed_base_cased

[https://github.com/naver/biobert-pretrained](https://github.com/naver/biobert-pretrained)

{:.h2_title}
## Data Source
Trained on PubMed + MIMIC III corpora

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobert_pubmed_base_cased_en_2.6.0_2.4_1600182457870.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = BertEmbeddings.pretrained("biobert_pubmed_base_cased","en","clinical/models") \
	.setInputCols("document","sentence","token") \
	.setOutputCol("word_embeddings")
```

```scala
val model = BertEmbeddings.pretrained("biobert_pubmed_base_cased","en","clinical/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|---------------------------|
| Model Name     | biobert_pubmed_base_cased |
| Type           | BertEmbeddings            |
| Compatibility  | 2.5.0                     |
| License        | Licensed                  |
| Edition        | Healthcare                |
| Inputs         | document, sentence, token |
| Output         | word_embeddings           |
| Language       | en                        |
| Dimension      | 768                       |
| Case Sensitive | True                      |

