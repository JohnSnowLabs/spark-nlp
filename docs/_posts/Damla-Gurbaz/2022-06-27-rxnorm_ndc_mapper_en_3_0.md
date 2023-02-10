---
layout: model
title: Mapping RxNorm Codes with Corresponding National Drug Codes(NDC)
author: John Snow Labs
name: rxnorm_ndc_mapper
date: 2022-06-27
tags: [rxnorm, ndc, chunk_mapper, licensed, clinical, en]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_ndc_mapper_en_3.5.3_3.0_1656314699115.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_ndc_mapper_en_3.5.3_3.0_1656314699115.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")\
.setInputCols(["ner_chunk", "sbert_embeddings"])\
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel\
.pretrained("rxnorm_ndc_mapper", "en", "clinical/models")\
.setInputCols(["rxnorm_code"])\
.setOutputCol("ndc_mappings")\
.setRels(["Product NDC", "Package NDC"])


pipeline = Pipeline(stages = [
documentAssembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text")) 

light_pipeline = LightPipeline(model)

result = light_pipeline.annotate(["doxycycline hyclate 50 MG Oral Tablet", "macadamia nut 100 MG/ML"])
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols("ner_chunk")
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel
.pretrained("rxnorm_ndc_mapper", "en", "clinical/models")
.setInputCols("rxnorm_code")
.setOutputCol("ndc_mappings")
.setRels(["Product NDC", "Package NDC"])

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper
))

val data = Seq(Array("doxycycline hyclate 50 MG Oral Tablet", "macadamia nut 100 MG/ML")).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.rxnorm_to_ndc").predict("""doxycycline hyclate 50 MG Oral Tablet""")
```

</div>

## Results

```bash
|    | ner_chunk                             |   rxnorm_code | Package NDC   | Product NDC   |
|---:|:--------------------------------------|--------------:|:--------------|:--------------|
|  0 | doxycycline hyclate 50 MG Oral Tablet |       1652674 | 62135-0625-60 | 46708-0499    |
|  1 | macadamia nut 100 MG/ML               |        259934 | 13349-0010-39 | 13349-0010    |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_ndc_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|2.0 MB|
