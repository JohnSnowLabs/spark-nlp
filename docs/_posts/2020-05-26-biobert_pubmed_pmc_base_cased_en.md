---
layout: model
title: Biobert Embeddings (Pubmed Pmc Base Cased)
author: John Snow Labs
name: biobert_pubmed_pmc_base_cased
class: BertEmbeddings
language: en
repository: clinical/models
date: 2020-05-26
tags: [clinical,embeddings,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture
Contextual embeddings representation using biobert_pubmed_pmc_base_cased

## Predicted Entities 
Contextual feature vectors based on biobert_pubmed_pmc_base_cased

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobert_pubmed_pmc_base_cased_en_2.5.0_2.4_1590489367180.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = BertEmbeddings.pretrained("biobert_pubmed_pmc_base_cased","en","clinical/models")\
	.setInputCols("document","sentence","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = BertEmbeddings.pretrained("biobert_pubmed_pmc_base_cased","en","clinical/models")
	.setInputCols("document","sentence","token")
	.setOutputCol("word_embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|----------------|-------------------------------|
| Name:           | biobert_pubmed_pmc_base_cased |
| Type:    | BertEmbeddings                |
| Compatibility:  | Spark NLP 2.5.0+                         |
| License:        | Licensed                      |
|Edition:|Official|                    |
|Input labels:         | [document, sentence, token]     |
|Output labels:        | [word_embeddings]               |
| Language:       | en                            |
| Dimension:     | 768.0                         |
| Case sensitive: | True                          |

{:.h2_title}
## Data Source
Trained on PubMed + MIMIC III corpora
https://github.com/naver/biobert-pretrained