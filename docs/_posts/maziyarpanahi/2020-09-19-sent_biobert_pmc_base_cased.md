---
layout: model
title: BioBERT Sentence Embeddings (PMC)
author: John Snow Labs
name: sent_biobert_pmc_base_cased
date: 2020-09-19
task: Embeddings
language: en
edition: Spark NLP 2.6.2
tags: [embeddings, en, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a pre-trained weights of BioBERT, a language representation model for biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. The details are described in the paper "[BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_biobert_pmc_base_cased_en_2.6.2_2.4_1600532770743.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

embeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pmc_base_cased", "en") \
      .setInputCols("sentence") \
      .setOutputCol("sentence_embeddings")
```

```scala

val embeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pmc_base_cased", "en")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_biobert_pmc_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.6.2|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|true|


{:.h2_title}
## Data Source
The model is imported from [https://github.com/dmis-lab/biobert](https://github.com/dmis-lab/biobert)
