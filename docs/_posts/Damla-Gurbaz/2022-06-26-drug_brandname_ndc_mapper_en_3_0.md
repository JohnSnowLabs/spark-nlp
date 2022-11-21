---
layout: model
title: Mapping Drug Brand Names with Corresponding National Drug Codes
author: John Snow Labs
name: drug_brandname_ndc_mapper
date: 2022-06-26
tags: [chunk_mapper, ndc, clinical, licensed, en]
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

This pretrained model maps drug brand names to corresponding National Drug Codes (NDC). Product NDCs for each strength are returned in result and metadata.

## Predicted Entities

`Strength_NDC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/drug_brandname_ndc_mapper_en_3.5.3_3.0_1656260706121.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("chunk")

chunkerMapper = ChunkMapperModel.pretrained("drug_brandname_ndc_mapper", "en", "clinical/models")\
.setInputCols(["chunk"])\
.setOutputCol("ndc")\
.setRels(["Strength_NDC"])\
.setLowerCase(True)


pipeline = Pipeline().setStages([
			document_assembler,
			chunkerMapper])  


model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text")) 

light_pipeline = LightPipeline(model)

result = light_pipeline.fullAnnotate(["zytiga", "zyvana", "ZYVOX"])
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("chunk")

val chunkerMapper = ChunkMapperModel.pretrained("drug_brandname_ndc_mapper", "en", "clinical/models")
.setInputCols(Array("chunk"))
.setOutputCol("ndc")
.setRels(Array("Strength_NDC"))
.setLowerCase(True)


val pipeline = new Pipeline().setStages(Array(
				  document_assembler,
				  chunkerMapper))

val sample_data = Seq("zytiga", "zyvana", "ZYVOX").toDS.toDF("text")

val result = pipeline.fit(sample_data).transform(sample_data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.drug_brand_to_ndc").predict("""Put your text here.""")
```

</div>

## Results

```bash
|    | Brandname   | Strength_NDC             |
|---:|:------------|:-------------------------|
|  0 | zytiga      | 500 mg/1 | 57894-195     |
|  1 | zyvana      | 527 mg/1 | 69336-405     |
|  2 | ZYVOX       | 600 mg/300mL | 0009-4992 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drug_brandname_ndc_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|3.0 MB|
