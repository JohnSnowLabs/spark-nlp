---
layout: model
title: Chinese BertForTokenClassification Base Cased model (from ckiplab)
author: John Snow Labs
name: bert_token_classifier_base_han_chinese_pos_shanggu
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-han-chinese-pos-shanggu` is a Chinese model originally trained by `ckiplab`.

## Predicted Entities

`VH1n`, `VPYU`, `VH2`, `VFYW`, `VKw`, `NA3`, `VJW`, `VC2UXZ`, `VC1UZ`, `VC1UX`, `DL`, `dD`, `VJU`, `VPWY`, `VGXY`, `VGYZ`, `VFWZ`, `VDYX`, `VEP`, `VC1Xw`, `s`, `N`, `VFWZU`, `VFXZU`, `VKXWZ`, `VJY`, `VFOU`, `VPXWZ`, `VDXW`, `VC1ｙ`, `VEWU`, `VPXWU`, `VC1X`, `VC1XZU`, `VGU`, `FW`, `VFY`, `VC2YX`, `VC2z`, `VD`, `A`, `VC1XZ`, `VKXW`, `VEXU`, `VC1WZ`, `VC2XZ`, `VF`, `VPXYW`, `VAUZ`, `VKXY`, `Vh1`, `DJ`, `VC1YXW`, `VEYZ`, `b`, `VKXUZ`, `VKWXZ`, `VC1XYU`, `VH1UN`, `VKXw`, `VJOU`, `VC1XW`, `VFW`, `VC2Z`, `VJWY`, `Vc1`, `VC2XZU`, `VKW`, `VAU`, `VKWZ`, `VC2XU`, `VE`, `I`, `VPZ`, `VC2YWZ`, `VC1I`, `ETCCATEGORY`, `VPYY`, `VGXZ`, `VC`, `Dc`, `VDOZ`, `NH`, `VFXZ`, `VGYW`, `NA2`, `VFXU`, `VKOZ`, `VC2X`, `VC1ZO`, `VKXO`, `VKXYW`, `NA4`, `VC1XWUZ`, `VDXZ`, `NB5`, `VJXWZ`, `VC1XWU`, `VDWU`, `SHI`, `VKWUZ`, `VC1W`, `VFWY`, `QUESTIONCATEGORY`, `c`, `VC1YXU`, `VPW`, `DD`, `VJWO`, `VC2`, `VC2XWU`, `VC1`, `VG`, `T`, `VEX`, `VEWX`, `VH2S`, `VFYZ`, `VC1ZY`, `VC1Ou`, `Dv`, `VKUXZ`, `VJWZ`, `VEXWU`, `VC1w`, `VI`, `Na1`, `VC1U`, `VEW`, `VC2UZ`, `VHIS`, `VKXWU`, `VGZ`, `VC2XWZ`, `NII`, `BU`, `Na4`, `VFXW`, `VEXWY`, `VC1YZ`, `VPXw`, `VC2W`, `VKXZU`, `VH`, `Nb2`, `B2`, `VH2US`, `VAN`, `NIU`, `VEO`, `VC2WX`, `VEZU`, `VDYZ`, `VPUO`, `VPXZU`, `VPO`, `VC1UN`, `VC1WZU`, `VC1XYW`, `97`, `VFUXZ`, `VH2U`, `VC1XY`, `VJUX`, `VDZ`, `VFU`, `VC1XU`, `VDZU`, `VFXWZ`, `Ng`, `VH2Y`, `VC1XWY`, `VC2OU`, `VKYW`, `NI`, `NB4`, `VC1XWZ`, `VJX`, `VPYZ`, `VJ`, `VFWU`, `VFYU`, `VEXY`, `VC1Z`, `VC1OUZ`, `VKO`, `VFUZ`, `VEZY`, `VC2UY`, `VDU`, `VEWY`, `Dl`, `VKUZ`, `SEMICOLONCATEGOR`, `VC1WU`, `VJYW`, `VKWW`, `VC2XY`, `VC1O`, `VDOU`, `VC1YY`, `VPYW`, `VJXWY`, `VJXYW`, `VPXYZ`, `VH1U`, `VC2WU`, `VGW`, `S`, `VKXWY`, `VO`, `VKZU`, `PERIODCATEGORY`, `VDY`, `VC2XYW`, `Na2`, `VFZ`, `VC2UO`, `dC`, `COLONCATEGORY`, `VJZU`, `VFOZ`, `EXCLAMATIONCATEGORY`, `VKZX`, `VKY`, `VM`, `VAW`, `VC2XWY`, `VJYU`, `VPYX`, `VAS`, `VC1ｘ`, `VEY`, `VJXW`, `U`, `VKOU`, `VC1XX`, `VC1YW`, `VEUX`, `VKYXW`, `VGX`, `VJXZU`, `VC1UY`, `NE`, `Da`, `VCY`, `t`, `VDYXU`, `VFXWY`, `VEXw`, `VPWU`, `VC1WX`, `VKWY`, `VKXU`, `VEZ`, `NA`, `DH`, `VEWZ`, `VPXW`, `VPOZ`, `VPWX`, `VDWX`, `VH1NU`, `VC2WZ`, `VJWU`, `VGY`, `VDXWU`, `VEOZ`, `VKZY`, `VC2w`, `VC1OZ`, `Db`, `VKU`, `SEMICOLONCATEGORY`, `Vp`, `VKUO`, `VC1YX`, `VC2YW`, `VGWY`, `VDXY`, `Ve`, `VEZZ`, `VPWZ`, `VC1OZU`, `Vd`, `VEU`, `VJWZU`, `VJXY`, `VPｏ`, `Nh`, `VC2WZU`, `VH1WU`, `NF`, `VKXZ`, `VH11`, `VJZ`, `VGP`, `V`, `VC2U`, `VKUX`, `NA5`, `VPOU`, `VGWX`, `VC1x`, `VKYXZ`, `VEUW`, `p`, `VC1Y`, `VJXZ`, `VC1UO`, `VH2SU`, `COMMACATEGORY`, `VAZ`, `VH12`, `Va`, `VFw`, `VKX`, `VKXYZ`, `VJYX`, `VFYX`, `VJYZ`, `VPU`, `VC12`, `VC1YU`, `MH`, `VFUO`, `VH12N`, `VFZU`, `VH1S`, `VC2Y`, `VJYY`, `C`, `VPUZ`, `VFX`, `VKOUZ`, `VA`, `NG`, `VC1XUZ`, `VH1ZU`, `DB`, `VPY`, `PAUSECATEGORY`, `VDX`, `VEXW`, `NB3`, `VAn`, `VPXY`, `DK`, `VFO`, `VDXWZ`, `D`, `VGO`, `NB2`, `DG`, `VC1OUN`, `VDOUZ`, `VFUY`, `VH1N`, `DA`, `VEXZ`, `VC2ZU`, `DN`, `VFXWU`, `NB1`, `VC2O`, `VJWX`, `VH2N`, `VC2YZ`, `VPXWY`, `VEYX`, `DC`, `VGz`, `VPXZ`, `VDUZ`, `VDXU`, `VEUZ`, `VP`, `P`, `VC1UW`, `VEYW`, `vF`, `EXCLANATIONCATEGORY`, `H1`, `VJUZ`, `VEE`, `VDO`, `DV`, `VC1WXU`, `VKZ`, `vG`, `NAN`, `VKWU`, `VC1ZU`, `nI`, `DF`, `VDW`, `NA1`, `VGYX`, `VAZU`, `VEOU`, `VPX`, `VFI`, `PARENTHESISCATEGORY`, `VPZU`, `VEXWZ`, `VK`, `VANU`, `VGXWY`, `VC2OZ`, `VEZX`, `VJO`, `VE0`, `VC1OU`, `VKYZ`, `VGXW`, `VEXZU`, `VJXU`, `VExW`, `VKWX`, `VC1WY`, `VC2XW`, `VC2YXW`, `VH1`, `VDXZU`, `VFXY`, `VPXU`, `VKYX`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_shanggu_zh_4.3.1_3.0_1679333600434.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_shanggu_zh_4.3.1_3.0_1679333600434.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos_shanggu","zh") \
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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos_shanggu","zh") 
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
|Model Name:|bert_token_classifier_base_han_chinese_pos_shanggu|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|397.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ckiplab/bert-base-han-chinese-pos-shanggu
- https://github.com/ckiplab/han-transformers
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/dkiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/pkiwi/kiwi.sh
- http://asbc.iis.sinica.edu.tw
- https://ckip.iis.sinica.edu.tw/