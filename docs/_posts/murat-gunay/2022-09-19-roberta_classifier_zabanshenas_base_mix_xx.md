---
layout: model
title: English RoBertaForSequenceClassification Base Cased model (from m3hrdadfi)
author: John Snow Labs
name: roberta_classifier_zabanshenas_base_mix
date: 2022-09-19
tags: [xx, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: xx
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `zabanshenas-roberta-base-mix` is a English model originally trained by `m3hrdadfi`.

## Predicted Entities

`mon`, `mdf`, `sun`, `bho`, `bxr`, `kaz`, `mrj`, `nld`, `dty`, `ben`, `mlt`, `arz`, `fur`, `pan`, `rup`, `ilo`, `srp`, `mwl`, `tat`, `mhr`, `som`, `vie`, `bjn`, `krc`, `mzn`, `nno`, `tur`, `bel`, `olo`, `mya`, `tam`, `pus`, `roh`, `ido`, `pdc`, `nds`, `ltg`, `lit`, `fas`, `kin`, `lao`, `lav`, `egl`, `lzh`, `afr`, `bod`, `map-bms`, `ina`, `pfl`, `wln`, `war`, `mri`, `ton`, `nap`, `hye`, `oci`, `new`, `gle`, `kbd`, `eng`, `nav`, `que`, `lug`, `cym`, `pol`, `sah`, `nds-nl`, `tuk`, `bul`, `chr`, `isl`, `ava`, `orm`, `scn`, `nan`, `azb`, `aym`, `slk`, `szl`, `wuu`, `sco`, `sgs`, `srd`, `mai`, `lad`, `amh`, `cdo`, `urd`, `nrm`, `por`, `cbk`, `san`, `sin`, `lrc`, `ukr`, `lez`, `vec`, `uig`, `ceb`, `tgl`, `glg`, `cat`, `pam`, `eus`, `chv`, `kir`, `nep`, `vol`, `est`, `dan`, `hsb`, `kor`, `nob`, `ara`, `ile`, `jam`, `srn`, `lat`, `zho`, `snd`, `epo`, `fry`, `swe`, `xmf`, `cos`, `bak`, `vls`, `ces`, `tel`, `ckb`, `zea`, `lim`, `nci`, `ron`, `lin`, `uzb`, `kat`, `aze`, `frp`, `hau`, `hbs`, `ibo`, `bpy`, `glv`, `heb`, `rus`, `kan`, `che`, `tsn`, `bcl`, `min`, `hat`, `fra`, `yid`, `kom`, `ast`, `ita`, `be-tarask`, `myv`, `tcy`, `lij`, `hak`, `sqi`, `gla`, `glk`, `sme`, `pap`, `mlg`, `ell`, `tha`, `hrv`, `tet`, `asm`, `als`, `crh`, `vep`, `pcd`, `sna`, `slv`, `diq`, `kur`, `dsb`, `jbo`, `ext`, `ind`, `yor`, `ori`, `mal`, `guj`, `grn`, `vro`, `spa`, `fin`, `cor`, `bre`, `nso`, `roa-tara`, `udm`, `tgk`, `jpn`, `hun`, `csb`, `bos`, `jav`, `bar`, `fao`, `ang`, `pag`, `hin`, `arg`, `stq`, `gag`, `hif`, `zh-yue`, `msa`, `kok`, `xho`, `koi`, `ltz`, `rue`, `wol`, `ace`, `kaa`, `lmo`, `swa`, `oss`, `kab`, `ksh`, `mkd`, `pnb`, `khm`, `deu`, `tyv`, `div`, `mar`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_zabanshenas_base_mix_xx_4.1.0_3.0_1663619837290.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_zabanshenas_base_mix","xx") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_zabanshenas_base_mix","xx") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_zabanshenas_base_mix|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|xx|
|Size:|416.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/m3hrdadfi/zabanshenas-roberta-base-mix
- https://github.com/m3hrdadfi/zabanshenas