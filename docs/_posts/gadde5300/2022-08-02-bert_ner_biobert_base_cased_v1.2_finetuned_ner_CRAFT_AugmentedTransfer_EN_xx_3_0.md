---
layout: model
title: Multilingual BertForTokenClassification Base Cased model (from StivenLancheros)
author: John Snow Labs
name: bert_ner_biobert_base_cased_v1.2_finetuned_ner_CRAFT_AugmentedTransfer_EN
date: 2022-08-02
tags: [bert, ner, open_source, en, es, xx]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `biobert-base-cased-v1.2-finetuned-ner-CRAFT_AugmentedTransfer_EN` is a Multilingual model originally trained by `StivenLancheros`.

## Predicted Entities

`GENE`, `Chemical`, `Protein`, `Cell`, `Sequence`, `Taxon`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_biobert_base_cased_v1.2_finetuned_ner_CRAFT_AugmentedTransfer_EN_xx_4.1.0_3.0_1659455698992.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_biobert_base_cased_v1.2_finetuned_ner_CRAFT_AugmentedTransfer_EN","xx") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_biobert_base_cased_v1.2_finetuned_ner_CRAFT_AugmentedTransfer_EN","xx") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_biobert_base_cased_v1.2_finetuned_ner_CRAFT_AugmentedTransfer_EN|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/StivenLancheros/biobert-base-cased-v1.2-finetuned-ner-CRAFT_AugmentedTransfer_EN