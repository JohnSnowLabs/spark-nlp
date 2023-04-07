---
layout: model
title: Chinese BertForTokenClassification Base Cased model (from ckiplab)
author: John Snow Labs
name: bert_token_classifier_base_han_chinese_pos_jindai
date: 2023-03-20
tags: [zh, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: zh
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-han-chinese-pos-jindai` is a Chinese model originally trained by `ckiplab`.

## Predicted Entities

`T6`, `SHI`, `VAL`, `Va`, `Dk`, `VAC`, `QUESTIONCATEGORY`, `Dj`, `/Na`, `Dba`, `DASHCATEGORY`, `Daa`, `Df`, `PERIODCATEGORY`, `VG`, `T`, `Q`, `Neqa`, `T3`, `Vf`, `COLONCATEGORY`, `VU`, `VI`, `EXCLAMATIONCATEGORY`, `VA`, `DE`, `Dd`, `R`, `u`, `Nf`, `N`, `PAUSECATEGORY`, `3`, `Dfb`, `q`, `VHL`, `Nd`, `Dfa`, `D`, `T4`, `x`, `FW`, `坐`, `Na`, `Vk`, `A`, `cr`, `VD`, `T7`, `VCL`, `V_2`, `Dbb`, `COMMACATEGORY`, `VF`, `VCl`, `Vh`, `VH`, `NA`, `PARENTHESISCATEGOR`, `VB`, `CE`, `b`, `V-2`, `P`, `Nes`, `EXCLANATIONCATEGORY`, `SEMICOLONCATEGORY`, `Di`, `Dab`, `VL`, `I`, `VE`, `Ng`, `ETCCATEGORY`, `Dh`, `VC`, `Caa`, `Nh`, `Dg`, `PARENTHESISCATEGORY`, `Dc`, `籙`, `VK`, `VJ`, `Neu`, `V`, `Dha`, `Nb`, `Dl`, `T5`, `Nc`, `Cbb`, `X`, `Nha`, `VHC`, `T8`, `U`, `V_`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_jindai_zh_4.3.1_3.0_1679333562988.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_jindai_zh_4.3.1_3.0_1679333562988.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos_jindai","zh") \
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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos_jindai","zh") 
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
|Model Name:|bert_token_classifier_base_han_chinese_pos_jindai|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|396.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ckiplab/bert-base-han-chinese-pos-jindai
- https://github.com/ckiplab/han-transformers
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/dkiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/pkiwi/kiwi.sh
- http://asbc.iis.sinica.edu.tw
- https://ckip.iis.sinica.edu.tw/