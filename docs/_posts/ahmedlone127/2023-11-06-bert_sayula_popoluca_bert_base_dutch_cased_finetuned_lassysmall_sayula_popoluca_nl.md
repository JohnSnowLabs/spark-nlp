---
layout: model
title: Dutch, Flemish bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca BertForTokenClassification from wietsedv
author: John Snow Labs
name: bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca
date: 2023-11-06
tags: [bert, nl, open_source, token_classification, onnx]
task: Named Entity Recognition
language: nl
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca` is a Dutch, Flemish model originally trained by wietsedv.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca_nl_5.2.0_3.0_1699303346752.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca_nl_5.2.0_3.0_1699303346752.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = BertForTokenClassification.pretrained("bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca","nl") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val documentAssembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val tokenClassifier = BertForTokenClassification  
    .pretrained("bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca", "nl")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sayula_popoluca_bert_base_dutch_cased_finetuned_lassysmall_sayula_popoluca|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|nl|
|Size:|407.3 MB|

## References

https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-lassysmall-pos