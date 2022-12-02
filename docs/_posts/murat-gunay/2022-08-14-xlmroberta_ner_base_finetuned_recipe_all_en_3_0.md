---
layout: model
title: English XLMRobertaForTokenClassification Base Cased model (from edwardjross)
author: John Snow Labs
name: xlmroberta_ner_base_finetuned_recipe_all
date: 2022-08-14
tags: [en, open_source, xlm_roberta, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-base-finetuned-recipe-all` is a English model originally trained by `edwardjross`.

## Predicted Entities

`UNIT`, `DF`, `QUANTITY`, `TEMP`, `SIZE`, `NAME`, `STATE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_finetuned_recipe_all_en_4.1.0_3.0_1660447490206.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_base_finetuned_recipe_all","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")
    
ner_converter = NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk") 
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, token_classifier, ner_converter])

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
 
val token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_base_finetuned_recipe_all","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("document", "token', "ner"))
    .setOutputCol("ner_chunk")
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, token_classifier, ner_converter))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_base_finetuned_recipe_all|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|835.0 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/edwardjross/xlm-roberta-base-finetuned-recipe-all
- https://github.com/cosylabiiit/recipe-knowledge-mining
- https://arxiv.org/abs/2004.12184
- https://github.com/cosylabiiit/recipe-knowledge-mining
- https://www.oreilly.com/library/view/natural-language-processing/9781098103231/
- https://github.com/EdwardJRoss/nlp_transformers_exercises/blob/master/notebooks/ch4-ner-recipe-stanford-crf.ipynb