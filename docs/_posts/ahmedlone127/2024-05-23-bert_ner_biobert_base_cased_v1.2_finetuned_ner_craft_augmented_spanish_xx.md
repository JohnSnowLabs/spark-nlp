---
layout: model
title: Multilingual bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish BertForTokenClassification from StivenLancheros
author: John Snow Labs
name: bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish
date: 2024-05-23
tags: [xx, open_source, onnx, token_classification, bert, ner]
task: Named Entity Recognition
language: xx
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish` is a Multilingual model originally trained by StivenLancheros.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish_xx_5.2.4_3.0_1716431525680.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish_xx_5.2.4_3.0_1716431525680.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier   = BertForTokenClassification.pretrained("bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish","xx") \
     .setInputCols(["token","document"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier  = BertForTokenClassification.pretrained("bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish", "xx")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmented_spanish|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|403.7 MB|

## References

https://huggingface.co/StivenLancheros/biobert-base-cased-v1.2-finetuned-ner-CRAFT_Augmented_ES