---
layout: model
title: BioBERT Clinical
author: John Snow Labs
name: biobert_clinical_base_cased
date: 2020-07-20
tags: [embeddings, en, bert]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a pre-trained weights of ClinicalBERT for generic clinical text. This domain-specific model has performance improvements on 3/5 clinical NLP tasks andd establishing a new state-of-the-art on the MedNLI dataset. The details are described in the paper "[Publicly Available Clinical BERT Embeddings](https://www.aclweb.org/anthology/W19-1909/)".

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_clinical_base_cased_en_2.5.0_2.4_1590489819943.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

embeddings = BertEmbeddings.pretrained("biobert_clinical_base_cased", "en") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```

```scala

val embeddings = BertEmbeddings.pretrained("biobert_clinical_base_cased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|biobert_clinical_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.5.0|
|License:|Open Source|
|Edition:|Official|
|Spark inputs:|[sentence, token]|
|Spark outputs:|[word_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|true|


{:.h2_title}
## Source
The model is imported from [https://github.com/EmilyAlsentzer/clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT)
