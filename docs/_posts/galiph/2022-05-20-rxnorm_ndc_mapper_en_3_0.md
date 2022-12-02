---
layout: model
title: Mapping RxNorm Codes with Corresponding National Drug Codes
author: John Snow Labs
name: rxnorm_ndc_mapper
date: 2022-05-20
tags: [ndc, rxnorm, en, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.1
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps RxNorm and RxNorm Extension codes with corresponding National Drug Codes (NDC).

## Predicted Entities

`Product NDC`, `Package NDC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_ndc_mapper_en_3.5.1_3.0_1653061064192.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
.setInputCol('text')\
.setOutputCol('ner_chunk')

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')
.setInputCols(["ner_chunk"])
.setOutputCol("sentence_embeddings")
.setCaseSensitive(False)

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models")
.setInputCols(["ner_chunk", "sentence_embeddings"])
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

chunkerMapper_product = ChunkMapperModel.pretrained("rxnorm_ndc_mapper", "en", "clinical/models"))
.setInputCols(["rxnorm_code"])
.setOutputCol("Product NDC")
.setRel("Product NDC")

chunkerMapper_package = ChunkMapperModel.pretrained("rxnorm_ndc_mapper", "en", "clinical/models"))
.setInputCols(["rxnorm_code"])
.setOutputCol("Package NDC")
.setRel("Package NDC")

pipeline = Pipeline().setStages([document_assembler, sbert_embedder, rxnorm_resolver, chunkerMapper_product, chunkerMapper_package ])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

lp = LightPipeline(model)

result = lp.annotate(['doxycycline hyclate 50 MG Oral Tablet', 'macadamia nut 100 MG/ML'])
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sentence_embeddings")
.setCaseSensitive(False)

val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper_product = ChunkMapperModel.pretrained("rxnorm_ndc_mapper", "en", "clinical/models"))
.setInputCols(Array("rxnorm_code"))
.setOutputCol("Product NDC")
.setRel("Product NDC")  

val chunkerMapper_package = ChunkMapperModel.pretrained("rxnorm_ndc_mapper", "en", "clinical/models"))
.setInputCols(Array("rxnorm_code"))
.setOutputCol("Package NDC")
.setRel("Package NDC") 

val pipeline = Pipeline().setStages(Array(document_assembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper_product,
chunkerMapper_package
))

val text_data = Seq("doxycycline hyclate 50 MG Oral Tablet'", "macadamia nut 100 MG/ML").toDF("text")
val res = pipeline.fit(text_data).transform(text_data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.rxnorm_to_ndc").predict("""Product NDC""")
```

</div>

## Results

```bash
|    | ner_chunk                                 | rxnorm_code   | Package NDC       | Product NDC    |
|---:|:------------------------------------------|:--------------|:------------------|:---------------|
|  0 | ['doxycycline hyclate 50 MG Oral Tablet'] | ['1652674']   | ['62135-0625-60'] | ['62135-0625'] |
|  1 | ['macadamia nut 100 MG/ML']               | ['212433']    | ['00187-1474-08'] | ['00187-1474'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_ndc_mapper|
|Compatibility:|Healthcare NLP 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|2.0 MB|
