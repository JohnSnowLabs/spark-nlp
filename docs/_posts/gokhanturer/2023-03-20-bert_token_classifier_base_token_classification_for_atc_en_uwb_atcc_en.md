---
layout: model
title: English BertForTokenClassification Base Cased model (from Jzuluaga)
author: John Snow Labs
name: bert_token_classifier_base_token_classification_for_atc_en_uwb_atcc
date: 2023-03-20
tags: [en, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-token-classification-for-atc-en-uwb-atcc` is a English model originally trained by `Jzuluaga`.

## Predicted Entities

`atco`, `pilot`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_token_classification_for_atc_en_uwb_atcc_en_4.3.1_3.0_1679332472572.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_token_classification_for_atc_en_uwb_atcc_en_4.3.1_3.0_1679332472572.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_token_classification_for_atc_en_uwb_atcc","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_token_classification_for_atc_en_uwb_atcc","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_base_token_classification_for_atc_en_uwb_atcc|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Jzuluaga/bert-base-token-classification-for-atc-en-uwb-atcc
- https://github.com/idiap/bert-text-diarization-atc
- https://arxiv.org/abs/2110.05781
- https://github.com/idiap/bert-text-diarization-atc
- https://arxiv.org/abs/2110.05781
- https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
- https://github.com/idiap/bert-text-diarization-atc/tree/main/data/databases/uwb_atcc
- https://github.com/idiap/bert-text-diarization-atc/blob/main/data/databases/uwb_atcc/data_prepare_uwb_atcc_corpus.sh
- https://github.com/idiap/bert-text-diarization-atc/blob/main/data/databases/uwb_atcc/exp_prepare_uwb_atcc_corpus.sh
- https://paperswithcode.com/sota?task=chunking&dataset=UWB-ATCC+corpus+%28Air+Traffic+Control+Communications%29