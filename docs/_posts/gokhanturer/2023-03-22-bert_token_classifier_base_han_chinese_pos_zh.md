---
layout: model
title: Chinese BertForTokenClassification Base Cased model (from ckiplab)
author: John Snow Labs
name: bert_token_classifier_base_han_chinese_pos
date: 2023-03-22
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-han-chinese-pos` is a Chinese model originally trained by `ckiplab`.

## Predicted Entities

`Vh`, `VEWX`, `QUESTIONCATEGORY`, `VPUZ`, `VC1XWUZ`, `VEUX`, `VJO`, `VEWZ`, `Nha`, `VC1WZU`, `NB1`, `VPUO`, `SEMICOLONCATEGORY`, `Dha`, `VC1WX`, `VGXZ`, `VKUZ`, `s`, `VEX`, `VDXWU`, `Na4`, `VC1WU`, `I`, `VPXWZ`, `VFXY`, `VKWU`, `NA`, `VJUX`, `Na`, `NB2`, `VEO`, `VJXW`, `VC1ZO`, `VFXW`, `VC2YWZ`, `VJUZ`, `Nb`, `VPXYW`, `VKWY`, `Dfb`, `SEMICOLONCATEGOR`, `VJ`, `VJXYW`, `VJZU`, `VJWU`, `Db`, `VEXU`, `nc`, `NII`, `VEZZ`, `VEE`, `VJYY`, `t`, `VPX`, `NB3`, `V＿２`, `VC12`, `VC1YX`, `VC2O`, `VH1n`, `VGX`, `PARENTHESISCATEGORY`, `VKXYZ`, `NIU`, `Daa`, `NB`, `Q`, `DOTCATEGORY`, `VJXY`, `VC1w`, `VC2YXW`, `VC2OZ`, `VKXWZ`, `VHC`, `VJXZU`, `DG`, `q`, `V_`, `COLONCATEGORY`, `VC1Y`, `Ne`, `VAL`, `Cba`, `D`, `VEUW`, `Ve`, `DB`, `T7`, `VH2N`, `VC1UZ`, `VKXUZ`, `Nv`, `VFXU`, `Dv`, `VFw`, `DF`, `VJY`, `VH2`, `T6`, `VC1OUZ`, `VDOU`, `VCl`, `VC1WZ`, `VPOU`, `Cab`, `VPWY`, `Ng`, `VDUZ`, `VJWO`, `DFa`, `VDYX`, `VC2YW`, `VJX`, `VDOZ`, `VEWY`, `VEXw`, `Dl`, `Dbb`, `VKXW`, `VGW`, `VH1N`, `VPYY`, `DC`, `DA`, `VFWU`, `VJOU`, `c`, `VEYW`, `VC2XZU`, `VC1UX`, `VGWX`, `VJWX`, `Vf`, `VC1UO`, `VFWZ`, `VKZU`, `VC2OU`, `VDXY`, `nI`, `R`, `VEXWZ`, `VANU`, `VGYW`, `sHI`, `VGZ`, `VGXY`, `VFYU`, `T5`, `VH`, `VC1XWU`, `/Na`, `VFUZ`, `T3`, `VEZU`, `VPYU`, `VC1XZ`, `dI`, `VC1OU`, `Vh1`, `VCL`, `VPYW`, `VC1WXU`, `ETCCATEGORY`, `VDYZ`, `NA1`, `vG`, `VPWZ`, `VKXO`, `Neqb`, `VFUY`, `VFU`, `T`, `VE`, `VC1UY`, `VFXWZ`, `VKXWY`, `VKXZU`, `Dc`, `VC2X`, `VEXWU`, `VC2Y`, `VKO`, `SHI`, `nf`, `VDXWZ`, `VEXWY`, `VKWZ`, `VPXw`, `VKXY`, `NA2`, `VJZ`, `COMMACATEGORY`, `dC`, `VKXw`, `DL`, `VH2U`, `VDXZU`, `VKYW`, `VD`, `Nd`, `VKYX`, `VKOU`, `VKXWU`, `VH11`, `VDW`, `VPU`, `VKUX`, `VAU`, `MH`, `VFOU`, `VC1YU`, `VC1Ou`, `VFYZ`, `DAb`, `VC2XZ`, `Dh`, `VC1W`, `VEWU`, `PERIODCATEGORY`, `VPOZ`, `VKZ`, `VC2WZU`, `r`, `VC2XWZ`, `VFI`, `VKY`, `Dfa`, `DASHCATEGORY`, `B2`, `VI`, `VKUXZ`, `VC1X`, `VC1XY`, `VC1ｙ`, `VEU`, `VKX`, `Nep`, `VJYU`, `VH1UN`, `VC2XY`, `VPWX`, `NI`, `VEP`, `VC2UO`, `VC1x`, `Df`, `VEXW`, `VFY`, `Dg`, `VPXZU`, `VC1WY`, `VDZ`, `VC1XUZ`, `VDXZ`, `Dab`, `VC2WU`, `na`, `VJYX`, `VEXZU`, `VC1OZ`, `VC1YY`, `Dba`, `VGY`, `VH1`, `VGz`, `VDOUZ`, `VPXZ`, `VFUXZ`, `VC2WZ`, `VKYXZ`, `VFW`, `NA4`, `Di`, `NH`, `VDXW`, `Vc1`, `VJXWY`, `VC1XYU`, `VJWY`, `VEOZ`, `VC1O`, `H1`, `VAW`, `PAUSECATEGORY`, `b`, `dD`, `VC1YXW`, `VC1YZ`, `VPZU`, `VFZU`, `VKWXZ`, `VO`, `ND`, `VExW`, `VFOZ`, `VH2US`, `VFYW`, `VC1Z`, `VDU`, `VKOUZ`, `VJXU`, `Da`, `VC2XWU`, `Vc`, `VHIS`, `VKWW`, `VFX`, `VC2`, `VEZ`, `VKXYW`, `vH`, `Vg`, `DH`, `VKXU`, `V`, `VFXWU`, `VKYXW`, `VC1XWY`, `DD`, `VKZY`, `VC1UW`, `Nb2`, `VH1WU`, `VC1XU`, `Na2`, `VC2U`, `EXCLANATIONCATEGORY`, `Ncd`, `P`, `VPO`, `VB`, `T4`, `VEZX`, `VF`, `NB5`, `NG`, `V_2`, `VH2S`, `VFYX`, `VC1OZU`, `VJU`, `VC2XW`, `VC1XZU`, `VPXYZ`, `VPｏ`, `VKWUZ`, `VL`, `VEXY`, `VK`, `VDXU`, `Vk`, `Vd`, `Vp`, `VPW`, `VC2UZ`, `VFXZ`, `VFWZU`, `Na1`, `VC2Z`, `VC1Xw`, `DK`, `VKYZ`, `NA5`, `Dk`, `VPYX`, `VC1ｘ`, `VC2YZ`, `NA3`, `VAZ`, `VP`, `VPXY`, `VJXWZ`, `VC2XYW`, `DJ`, `VH1NU`, `VJYW`, `De`, `NAN`, `VKUO`, `VPXW`, `Neqa`, `VPWU`, `VPZ`, `vF`, `VEYZ`, `V-2`, `Dj`, `VDO`, `VFUO`, `VC2XU`, `VKOZ`, `VC1I`, `VGXW`, `VC1OUN`, `VEOU`, `VEW`, `cr`, `VDWX`, `符，尚無資料`, `VHL`, `VC2WX`, `VJYZ`, `VH2Y`, `VAn`, `VC1`, `VPXWY`, `VGU`, `VC1U`, `NF`, `VJWZU`, `VJXZ`, `VJWZ`, `Neu`, `VC`, `Cbb`, `X`, `VFWY`, `VC2UXZ`, `VDY`, `VGWY`, `vC`, `VC1ZY`, `VE0`, `Nc`, `C`, `VC2z`, `EXCLAMATIONCATEGORY`, `VH1ZU`, `VEUZ`, `VFXZU`, `VPXWU`, `NE`, `VC2W`, `VDYXU`, `PARENTHES7ISCATEGORY`, `VDZU`, `VC1XYW`, `VDWU`, `VC2UY`, `VA`, `U`, `Dd`, `VC1ZU`, `VGYX`, `PARENTHESISCATEGOR`, `VAZU`, `N`, `VG`, `NB4`, `VC1YXU`, `VC1XX`, `VC1XWZ`, `VFXWY`, `VAC`, `VJW`, `VC2XWY`, `VFZ`, `VEZY`, `vA`, `VH2SU`, `VPY`, `S`, `DE`, `VKXZ`, `Va`, `DM`, `neu`, `CE`, `VFO`, `VEY`, `VH1U`, `A`, `VPXU`, `VH12`, `VC1YW`, `x`, `VGO`, `p`, `VEYX`, `Nes`, `VH1S`, `VC1XW`, `VAUZ`, `DN`, `VGYZ`, `坐`, `Caa`, `SPCHANGECATEGORY`, `BU`, `VKZX`, `FW`, `VPYZ`, `VM`, `VEXZ`, `VKW`, `VC1UN`, `VKWX`, `DV`, `VC2YX`, `VKw`, `VAS`, `籙`, `T8`, `VCY`, `Nf`, `VC2ZU`, `VH12N`, `Nh`, `VAN`, `VGP`, `VC2w`, `VKU`, `cbb`, `VU`, `u`, `VDX`, `VGXWY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_zh_4.3.1_3.0_1679492593842.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_han_chinese_pos_zh_4.3.1_3.0_1679492593842.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos","zh") \
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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_han_chinese_pos","zh") 
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
|Model Name:|bert_token_classifier_base_han_chinese_pos|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|397.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ckiplab/bert-base-han-chinese-pos
- https://github.com/ckiplab/han-transformers
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/dkiwi/kiwi.sh
- http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/pkiwi/kiwi.sh
- http://asbc.iis.sinica.edu.tw
- https://ckip.iis.sinica.edu.tw/