---
layout: model
title: English BertForSequenceClassification Cased model (from owaiskha9654)
author: John Snow Labs
name: bert_classifier_multi_label_classification_of_pubmed_articles
date: 2022-09-06
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Multi-Label-Classification-of-PubMed-Articles` is a English model originally trained by `owaiskha9654`.

## Predicted Entities

`Phenomena and Processes [G]`, `Anthropology, Education, Sociology, and Social Phenomena [I]`, `Diseases [C]`, `Health Care [N]`, `Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]`, `Chemicals and Drugs [D]`, `Psychiatry and Psychology [F]`, `Anatomy [A]`, `Information Science [L]`, `Geographicals [Z]`, `Organisms [B]`, `Technology, Industry, and Agriculture [J]`, `Disciplines and Occupations [H]`, `Named Groups [M]`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_multi_label_classification_of_pubmed_articles_en_4.1.0_3.0_1662500997655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_multi_label_classification_of_pubmed_articles","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_multi_label_classification_of_pubmed_articles","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_multi_label_classification_of_pubmed_articles|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/owaiskha9654/Multi-Label-Classification-of-PubMed-Articles
- https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification
- https://www.kaggle.com/code/owaiskhan9654/multi-label-classification-of-pubmed-articles
- https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification
- https://arxiv.org/abs/1706.03762
- https://arxiv.org/abs/1810.04805
- https://github.com/google-research/bert
- https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss