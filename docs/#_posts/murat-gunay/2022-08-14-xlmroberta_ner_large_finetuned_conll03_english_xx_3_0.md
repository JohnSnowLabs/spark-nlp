---
layout: model
title: Multilingual XLMRobertaForTokenClassification Large Cased model
author: John Snow Labs
name: xlmroberta_ner_large_finetuned_conll03_english
date: 2022-08-14
tags: [xx, open_source, xlm_roberta, ner]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-large-finetuned-conll03-english` is a Multilingual model originally trained by HuggingFace.

## Predicted Entities

`ORG`, `LOC`, `PER`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_large_finetuned_conll03_english_xx_4.1.0_3.0_1660453726099.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_large_finetuned_conll03_english","xx") \
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
 
val token_classifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_large_finetuned_conll03_english","xx") 
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
|Model Name:|xlmroberta_ner_large_finetuned_conll03_english|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|1.8 GB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/xlm-roberta-large-finetuned-conll03-english
- https://arxiv.org/abs/1911.02116
- https://arxiv.org/abs/1911.02116
- https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr
- https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr
- https://arxiv.org/abs/1911.02116
- https://aclanthology.org/2021.acl-long.330.pdf
- https://dl.acm.org/doi/pdf/10.1145/3442188.3445922
- https://arxiv.org/pdf/2008.03415.pdf
- https://arxiv.org/pdf/2008.03415.pdf
- https://arxiv.org/pdf/1911.02116.pdf
- https://arxiv.org/pdf/1911.02116.pdf
- https://mlco2.github.io/impact#compute
- https://arxiv.org/abs/1910.09700
- https://arxiv.org/pdf/1911.02116.pdf
- https://arxiv.org/pdf/1911.02116.pdf