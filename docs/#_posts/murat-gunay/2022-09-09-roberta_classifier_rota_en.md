---
layout: model
title: English RobertaForSequenceClassification Cased model (from rti-international)
author: John Snow Labs
name: roberta_classifier_rota
date: 2022-09-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `rota` is a English model originally trained by `rti-international`.

## Predicted Entities

`TRAFFICKING - OTHER CONTROLLED SUBSTANCES`, `INVASION OF PRIVACY`, `HABITUAL OFFENDER`, `LARCENY/THEFT - VALUE UNKNOWN`, `TAX LAW (FEDERAL ONLY)`, `MANSLAUGHTER - VEHICULAR`, `CONTROLLED SUBSTANCE - OFFENSE UNSPECIFIED`, `WEAPON OFFENSE`, `RAPE - FORCE`, `DRUG OFFENSES - VIOLATION/DRUG UNSPECIFIED`, `TRAFFIC OFFENSES - MINOR`, `FLIGHT TO AVOID PROSECUTION`, `BRIBERY AND CONFLICT OF INTEREST`, `KIDNAPPING`, `AUTO THEFT`, `RIOTING`, `PROPERTY OFFENSES - OTHER`, `EMBEZZLEMENT (FEDERAL ONLY)`, `CHILD ABUSE`, `HEROIN VIOLATION - OFFENSE UNSPECIFIED`, `BLACKMAIL/EXTORTION/INTIMIDATION`, `GRAND LARCENY - THEFT OVER $200`, `DRIVING UNDER INFLUENCE - DRUGS`, `EMBEZZLEMENT`, `FORGERY/FRAUD`, `POSSESSION/USE - MARIJUANA/HASHISH`, `STOLEN PROPERTY - TRAFFICKING`, `FORGERY (FEDERAL ONLY)`, `PROBATION VIOLATION`, `FRAUD (FEDERAL ONLY)`, `OFFENSES AGAINST COURTS, LEGISLATURES, AND COMMISSIONS`, `UNARMED ROBBERY`, `ARSON`, `COCAINE OR CRACK VIOLATION OFFENSE UNSPECIFIED`, `SIMPLE ASSAULT`, `DESTRUCTION OF PROPERTY`, `POSSESSION/USE - DRUG UNSPECIFIED`, `COUNTERFEITING (FEDERAL ONLY)`, `FORCIBLE SODOMY`, `RAPE - STATUTORY - NO FORCE`, `UNAUTHORIZED USE OF VEHICLE`, `POSSESSION/USE - OTHER CONTROLLED SUBSTANCES`, `TRAFFICKING - DRUG UNSPECIFIED`, `IMMIGRATION VIOLATIONS`, `VOLUNTARY/NONNEGLIGENT MANSLAUGHTER`, `DRIVING WHILE INTOXICATED`, `PETTY LARCENY - THEFT UNDER $200`, `HIT/RUN DRIVING - PROPERTY DAMAGE`, `MURDER`, `REGULATORY OFFENSES (FEDERAL ONLY)`, `FAMILY RELATED OFFENSES`, `POSSESSION/USE - HEROIN`, `PUBLIC ORDER OFFENSES - OTHER`, `DRIVING UNDER THE INFLUENCE`, `TRESPASSING`, `CONTRIBUTING TO DELINQUENCY OF A MINOR`, `ARMED ROBBERY`, `FELONY - UNSPECIFIED`, `UNSPECIFIED HOMICIDE`, `MARIJUANA/HASHISH VIOLATION - OFFENSE UNSPECIFIED`, `TRAFFICKING - COCAINE OR CRACK`, `COMMERCIALIZED VICE`, `TRAFFICKING - HEROIN`, `LIQUOR LAW VIOLATIONS`, `ASSAULTING PUBLIC OFFICER`, `JUVENILE OFFENSES`, `VIOLENT OFFENSES - OTHER`, `MISDEMEANOR UNSPECIFIED`, `HIT AND RUN DRIVING`, `CONTEMPT OF COURT`, `BURGLARY`, `MANSLAUGHTER - NON-VEHICULAR`, `PAROLE VIOLATION`, `DRUNKENNESS/VAGRANCY/DISORDERLY CONDUCT`, `STOLEN PROPERTY - RECEIVING`, `TRAFFICKING MARIJUANA/HASHISH`, `SEXUAL ASSAULT - OTHER`, `LEWD ACT WITH CHILDREN`, `POSSESSION/USE - COCAINE OR CRACK`, `OBSTRUCTION - LAW ENFORCEMENT`, `RACKETEERING/EXTORTION (FEDERAL ONLY)`, `AGGRAVATED ASSAULT`, `MORALS/DECENCY - OFFENSE`, `ESCAPE FROM CUSTODY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_rota_en_4.1.0_3.0_1662767232954.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_rota","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_rota","en") 
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
|Model Name:|roberta_classifier_rota|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/rti-international/rota
- https://github.com/RTIInternational/rota
- https://doi.org/10.5281/zenodo.4770492
- https://www.icpsr.umich.edu/web/NACJD/studies/30799/datadocumentation#
- https://web.archive.org/web/20201021001250/https://www.icpsr.umich.edu/web/pages/NACJD/guides/ncrp.html