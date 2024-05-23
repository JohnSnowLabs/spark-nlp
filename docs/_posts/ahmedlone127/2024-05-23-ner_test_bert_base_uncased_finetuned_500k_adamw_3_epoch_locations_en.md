---
layout: model
title: English ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations BertForTokenClassification from poodledude
author: John Snow Labs
name: ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations
date: 2024-05-23
tags: [bert, en, open_source, token_classification, onnx]
task: Named Entity Recognition
language: en
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations` is a English model originally trained by poodledude.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations_en_5.2.4_3.0_1716457621573.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations_en_5.2.4_3.0_1716457621573.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = BertForTokenClassification.pretrained("ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations","en") \
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
    .pretrained("ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations", "en")
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
|Model Name:|ner_test_bert_base_uncased_finetuned_500k_adamw_3_epoch_locations|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.2 MB|

## References

References

https://huggingface.co/poodledude/ner-test-bert-base-uncased-finetuned-500K-AdamW-3-epoch-locations