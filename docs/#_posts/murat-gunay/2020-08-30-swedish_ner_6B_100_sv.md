---
layout: model
title: Named Entity Recognition (NER) Model in Swedish (GloVe 6B 100)
author: John Snow Labs
name: swedish_ner_6B_100
date: 2020-08-30
task: Named Entity Recognition
language: sv
edition: Spark NLP 2.6.0
spark_version: 2.4
tags: [ner, sv, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Swedish NER is a Named Entity Recognition (or NER) model, meaning it annotates text to find features like the names of people, places, and organizations. This NER model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together. The model is trained with GloVe 6B 100 word embeddings, so be sure to use the same embeddings in the pipeline.

{:.h2_title}
## Predicted Entities 
Persons-`PER`, Locations-`LOC`, Organizations-`ORG`, Product-`PRO`, Date-`DATE`, Event-`EVENT`.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_SV/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/swedish_ner_6B_100_sv_2.6.0_2.4_1598810268071.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = WordEmbeddingsModel.pretrained("glove_100d") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")
ner_model = NerDLModel.pretrained("swedish_ner_6B_100", "sv") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([['William Henry Gates III (född 28 oktober 1955) är en amerikansk affärsmagnat, mjukvaruutvecklare, investerare och filantrop. Han är mest känd som medgrundare av Microsoft Corporation. Under sin karriär på Microsoft innehade Gates befattningar som styrelseordförande, verkställande direktör (VD), VD och programvaruarkitekt samtidigt som han var den största enskilda aktieägaren fram till maj 2014. Han är en av de mest kända företagarna och pionjärerna inom mikrodatorrevolutionen på 1970- och 1980-talet. Född och uppvuxen i Seattle, Washington, grundade Gates Microsoft tillsammans med barndomsvän Paul Allen 1975 i Albuquerque, New Mexico; det blev vidare världens största datorprogramföretag. Gates ledde företaget som styrelseordförande och VD tills han avgick som VD i januari 2000, men han förblev ordförande och blev chef för programvaruarkitekt. Under slutet av 1990-talet hade Gates kritiserats för sin affärstaktik, som har ansetts konkurrensbegränsande. Detta yttrande har upprätthållits genom många domstolsbeslut. I juni 2006 meddelade Gates att han skulle gå över till en deltidsroll på Microsoft och heltid på Bill & Melinda Gates Foundation, den privata välgörenhetsstiftelsen som han och hans fru, Melinda Gates, grundade 2000. Han överförde gradvis sina uppgifter till Ray Ozzie och Craig Mundie. Han avgick som styrelseordförande i Microsoft i februari 2014 och tillträdde en ny tjänst som teknologrådgivare för att stödja den nyutnämnda VD Satya Nadella.']], ["text"]))
```

```scala
...
val embeddings = WordEmbeddingsModel.pretrained("glove_100d")
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")
val ner_model = NerDLModel.pretrained("swedish_ner_6B_100", "sv")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter))

val data = Seq("William Henry Gates III (född 28 oktober 1955) är en amerikansk affärsmagnat, mjukvaruutvecklare, investerare och filantrop. Han är mest känd som medgrundare av Microsoft Corporation. Under sin karriär på Microsoft innehade Gates befattningar som styrelseordförande, verkställande direktör (VD), VD och programvaruarkitekt samtidigt som han var den största enskilda aktieägaren fram till maj 2014. Han är en av de mest kända företagarna och pionjärerna inom mikrodatorrevolutionen på 1970- och 1980-talet. Född och uppvuxen i Seattle, Washington, grundade Gates Microsoft tillsammans med barndomsvän Paul Allen 1975 i Albuquerque, New Mexico; det blev vidare världens största datorprogramföretag. Gates ledde företaget som styrelseordförande och VD tills han avgick som VD i januari 2000, men han förblev ordförande och blev chef för programvaruarkitekt. Under slutet av 1990-talet hade Gates kritiserats för sin affärstaktik, som har ansetts konkurrensbegränsande. Detta yttrande har upprätthållits genom många domstolsbeslut. I juni 2006 meddelade Gates att han skulle gå över till en deltidsroll på Microsoft och heltid på Bill & Melinda Gates Foundation, den privata välgörenhetsstiftelsen som han och hans fru, Melinda Gates, grundade 2000. Han överförde gradvis sina uppgifter till Ray Ozzie och Craig Mundie. Han avgick som styrelseordförande i Microsoft i februari 2014 och tillträdde en ny tjänst som teknologrådgivare för att stödja den nyutnämnda VD Satya Nadella.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
text = ["""William Henry Gates III (född 28 oktober 1955) är en amerikansk affärsmagnat, mjukvaruutvecklare, investerare och filantrop. Han är mest känd som medgrundare av Microsoft Corporation. Under sin karriär på Microsoft innehade Gates befattningar som styrelseordförande, verkställande direktör (VD), VD och programvaruarkitekt samtidigt som han var den största enskilda aktieägaren fram till maj 2014. Han är en av de mest kända företagarna och pionjärerna inom mikrodatorrevolutionen på 1970- och 1980-talet. Född och uppvuxen i Seattle, Washington, grundade Gates Microsoft tillsammans med barndomsvän Paul Allen 1975 i Albuquerque, New Mexico; det blev vidare världens största datorprogramföretag. Gates ledde företaget som styrelseordförande och VD tills han avgick som VD i januari 2000, men han förblev ordförande och blev chef för programvaruarkitekt. Under slutet av 1990-talet hade Gates kritiserats för sin affärstaktik, som har ansetts konkurrensbegränsande. Detta yttrande har upprätthållits genom många domstolsbeslut. I juni 2006 meddelade Gates att han skulle gå över till en deltidsroll på Microsoft och heltid på Bill & Melinda Gates Foundation, den privata välgörenhetsstiftelsen som han och hans fru, Melinda Gates, grundade 2000. Han överförde gradvis sina uppgifter till Ray Ozzie och Craig Mundie. Han avgick som styrelseordförande i Microsoft i februari 2014 och tillträdde en ny tjänst som teknologrådgivare för att stödja den nyutnämnda VD Satya Nadella."""]

ner_df = nlu.load('sv.ner.6B_100').predict(text, output_level = "chunk")
ner_df[["entities", "entities_confidence"]]
```
</div>

{:.h2_title}
## Results

```bash
+---------------------+---------+
|chunk                |ner_label|
+---------------------+---------+
|William Henry Gates  |PER      |
|Microsoft Corporation|ORG      |
|Microsoft            |ORG      |
|Gates                |PER      |
|VD                   |MISC     |
|Seattle              |LOC      |
|Washington           |LOC      |
|Gates Microsoft      |PER      |
|Paul Allen           |PER      |
|Albuquerque          |LOC      |
|New Mexico           |ORG      |
|Gates                |PER      |
|Gates                |PER      |
|Microsoft            |ORG      |
|Melinda Gates        |PER      |
|Melinda Gates        |PER      |
|Ray Ozzie            |PER      |
|Craig Mundie         |PER      |
|Microsoft            |ORG      |
|VD Satya Nadella     |MISC     |
+---------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|swedish_ner_6B_100|
|Type:|ner|
|Compatibility:| Spark NLP 2.6.0+|
|Edition:|Official|
|License:|Open Source|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|sv|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on a custom dataset with multi-lingual GloVe Embeddings ``glove_6B_100``.