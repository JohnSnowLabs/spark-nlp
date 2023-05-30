---
layout: model
title: Chinese BertForTokenClassification Base Cased model (from ckiplab)
author: John Snow Labs
name: bert_token_classifier_base_han_chinese_pos_zhonggu
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-han-chinese-pos-zhonggu` is a Chinese model originally trained by `ckiplab`.

## Predicted Entities

`T6`, `SHI`, `Dk`, `VAC`, `NB`, `QUESTIONCATEGORY`, `Dj`, `Dba`, `DASHCATEGORY`, `nc`, `ND`, `Daa`, `Df`, `V＿２`, `PERIODCATEGORY`, `VG`, `T`, `Q`, `Vg`, `PARENTHES7ISCATEGORY`, `r`, `C`, `Neqa`, `T3`, `571`, `Vf`, `COLONCATEGORY`, `VI`, `EXCLAMATIONCATEGORY`, `VA`, `DE`, `Dd`, `R`, `Nf`, `N`, `PAUSECATEGORY`, `Dfb`, `Nd`, `Dfa`, `D`, `T4`, `FW`, `Na`, `VD`, `A`, `VCL`, `T7`, `Da`, `V_2`, `Dbb`, `VF`, `Ne`, `VH`, `NA`, `DH`, `DJ`, `DFa`, `VB`, `DC`, `b`, `P`, `Nes`, `EXCLANATIONCATEGORY`, `Db`, `SEMICOLONCATEGORY`, `Di`, `Dab`, `VL`, `neu`, `Ve`, `Vc`, `DAb`, `I`, `VE`, `na`, `Ng`, `Dh`, `VC`, `Caa`, `Nh`, `Dg`, `PARENTHESISCATEGORY`, `Dc`, `NH`, `VK`, `VJ`, `符，尚無資料`, `Neu`, `V`, `Nb`, `Dl`, `T5`, `Nc`, `Cbb`, `VHC`, `T8`, `U`, `COMMACATEGORY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_zhonggu_zh_4.3.1_3.0_1679333640183.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_zhonggu_zh_4.3.1_3.0_1679333640183.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos_zhonggu","zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos_zhonggu","zh")
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
|Model Name:|bert_token_classifier_base_han_chinese_pos_zhonggu|
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

- https://huggingface.co/ckiplab/bert-base-han-chinese-pos-zhonggu
- https://github.com/ckiplab/han-transformers
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/dkiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/pkiwi/kiwi.sh
- http://asbc.iis.sinica.edu.tw
- https://ckip.iis.sinica.edu.tw/