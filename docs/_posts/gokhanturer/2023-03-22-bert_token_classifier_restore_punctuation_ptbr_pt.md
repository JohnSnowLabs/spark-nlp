---
layout: model
title: Portuguese BertForTokenClassification Cased model (from dominguesm)
author: John Snow Labs
name: bert_token_classifier_restore_punctuation_ptbr
date: 2023-03-22
tags: [pt, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: pt
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-restore-punctuation-ptbr` is a Portuguese model originally trained by `dominguesm`.

## Predicted Entities

`.U`, `!O`, `,O`, `:O`, `,U`, `:U`, `;O`, `OU`, `?U`, `!U`, `OO`, `.O`, `-O`, `'O`, `?O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_restore_punctuation_ptbr_pt_4.3.1_3.0_1679492889197.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_restore_punctuation_ptbr_pt_4.3.1_3.0_1679492889197.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_restore_punctuation_ptbr","pt") \
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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_restore_punctuation_ptbr","pt") 
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
|Model Name:|bert_token_classifier_restore_punctuation_ptbr|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|406.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/dominguesm/bert-restore-punctuation-ptbr
- https://wandb.ai/dominguesm/RestorePunctuationPTBR
- https://github.com/DominguesM/respunct
- https://github.com/esdurmus/Wikilingua
- https://paperswithcode.com/sota?task=named-entity-recognition&dataset=wiki_lingua