---
layout: model
title: Icelandic RoBertaForSequenceClassification Cased model (from ueb1)
author: John Snow Labs
name: roberta_classifier_icebert_finetuned_grouped
date: 2022-09-19
tags: [is, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: is
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `IceBERT-finetuned-grouped` is a Icelandic model originally trained by `ueb1`.

## Predicted Entities

`Hulda Hólmkelsdóttir`, `Hjörvar Ólafsson`, `Birgir Þór Harðarson`, `Ásrún Brynja Ingvarsdóttir`, `Jóhanna María Einarsdóttir`, `Henry Birgir Gunnarsson`, `Kristinn Ingi Jónsson`, `Kristinn Ásgeir Gylfason`, `Birkir Blær Ingólfsson`, `Victor Pálsson`, `Magnús H. Jónasson`, `Kristín Ólafsdóttir`, `Jón Þór Kristjánsson`, `Jóhann Óli Eiðsson`, `Jón Júlíus Karlsson`, `Þorkell Gunnar Sigurbjörnsson`, `Garðar Örn Úlfarsson`, `Berghildur Erla Bernharðsdóttir`, `Oddur Ævar Gunnarsson`, `Kristján Róbert Kristjánsson`, `Kristinn Haukur Guðnason`, `Þórunn Elísabet Bogadóttir`, `Sylvía Rut Sigfúsdóttir`, `Hörður Snævar Jónsson`, `Finnur Thorlacius`, `Haukur Harðarson`, `Milla Ósk Magnúsdóttir`, `Kolbrún Bergþórsdóttir`, `Þórhildur Þorkelsdóttir`, `Kristín Sigurðardóttir`, `Magnús Hlynur Hreiðarsson`, `Tómas Þór Þórðarson`, `Einar Þór Sigurðsson`, `Erna Agnes Sigurgeirsdóttir`, `Alma Ómarsdóttir`, `Tryggvi Páll Tryggvason`, `Erla Dóra Magnúsdóttir`, `Birgir Olgeirsson`, `Máni Snær Þorláksson`, `Gunnar Hrafn Jónsson`, `Ingvi Þór Sæmundsson`, `Sylvía Hall`, `Róbert Jóhannsson`, `Arnar Geir Halldórsson`, `Þorsteinn Friðrik Halldórsson`, `Arnhildur Hálfdánardóttir`, `Eiríkur Stefán Ásgeirsson`, `Stígur Helgason`, `Valgerður Árnadóttir`, `Bjarni Pétur Jónsson`, `Óskar Ófeigur Jónsson`, `Andri Eysteinsson`, `Ingunn Lára Kristjánsdóttir`, `Einar Örn Jónsson`, `Ágúst Borgþór Sverrisson`, `Sigríður Dögg Auðunsdóttir`, `Jakob Bjarnar`, `Kolbeinn Tumi Daðason`, `Innanríkisráðuneyti`, `Bára Huld Beck`, `Ísak Hallmundarson`, `Valur Páll Eiríksson`, `Sunna Valgerðardóttir`, `Ingvar Þór Björnsson`, `Ævar Örn Jósepsson`, `Samúel Karl Ólason`, `Jón Hákon Halldórsson`, `Anna Lilja Þórisdóttir`, `Þórgnýr Einar Albertsson`, `Steindór Grétar Jónsson`, `Fjármálaráðuneyti`, `Haukur Holm`, `Forsætisráðuneyti`, `Hörður Ægisson`, `Sveinn Arnarsson`, `Margrét Helga Erlingsdóttir`, `Þorvaldur Friðriksson`, `Kristjana Arnarsdóttir`, `Atli Ísleifsson`, `Stefán Árni Pálsson`, `Sighvatur Arnmundsson`, `Anton Ingi Leifsson`, `Þórður Snær Júlíusson`, `Kristján Sigurjónsson`, `Rúnar Snær Reynisson`, `Karl Lúðvíksson`, `Birta Björnsdóttir`, `Jóhann K. Jóhannsson`, `Ari Brynjólfsson`, `Ólöf Ragnarsdóttir`, `Kristlín Dís Ingilínardóttir`, `Elín Margrét Böðvarsdóttir`, `Hólmfríður Dagný Friðjónsdóttir`, `Urður Örlygsdóttir`, `Jón Þór Stefánsson`, `Eiður Þór Árnason`, `Anna Kristín Jónsdóttir`, `Stefán Þór Hjartarson`, `Hallgrímur Indriðason`, `Ástrós Ýr Eggertsdóttir`, `Markús Þ. Þórhallsson`, `Freyr Gígja Gunnarsson`, `Kristján Már Unnarsson`, `Lovísa Arnardóttir`, `Rögnvaldur Már Helgason`, `Brynjólfur Þór Guðmundsson`, `Bergljót Baldursdóttir`, `Halla Ólafsdóttir`, `Úlla Árdal`, `Sólveig Klara Ragnarsdóttir`, `Gunnar Birgisson`, `Fanndís Birna Logadóttir`, `Sæunn Gísladóttir`, `Stefán Ó. Jónsson`, `Ásgeir Tómasson`, `Sunna Karen Sigurþórsdóttir`, `Runólfur Trausti Þórhallsson`, `Hallgerður Kolbrún E. Jónsdóttir`, `Katrín Ásmundsdóttir`, `Nadine Guðrún Yaghi`, `Andri Yrkill Valsson`, `Kjartan Kjartansson`, `Sindri Sverrisson`, `Jóhanna Vigdís Hjaltadóttir`, `Kristján Kristjánsson`, `Aðalheiður Ámundadóttir`, `Fókus`, `Jóhann Bjarni Kolbeinsson`, `Gunnþóra Gunnarsdóttir`, `Ágúst Ólafsson`, `Ása Ninna Pétursdóttir`, `Kristinn Páll Teitsson`, `Óttar Kolbeinsson Proppé`, `Kári Gylfason`, `Sunna Kristín Hilmarsdóttir`, `Þórdís Arnljótsdóttir`, `Þorvarður Pálsson`, `Guðrún Ósk Guðjónsdóttir`, `Heimir Már Pétursson`, `Vésteinn Örn Pétursson`, `Nína Hjördís Þorkelsdóttir`, `Dagný Hulda Erlendsdóttir`, `Magnús Halldórsson`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_icebert_finetuned_grouped_is_4.1.0_3.0_1663603111907.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_icebert_finetuned_grouped_is_4.1.0_3.0_1663603111907.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_icebert_finetuned_grouped","is") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_icebert_finetuned_grouped","is") 
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
|Model Name:|roberta_classifier_icebert_finetuned_grouped|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|is|
|Size:|464.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/ueb1/IceBERT-finetuned-grouped