---
layout: model
title: Sentence Entity Resolver for SNOMED codes (procedures and measurements)
author: John Snow Labs
name: sbiobertresolve_clinical_snomed_procedures_measurements
date: 2021-11-15
tags: [en, licensed, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.2
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps medical entities to SNOMED codes using `sent_biobert_clinical_base_cased` Sentence Bert Embeddings. The corpus of this model includes `Procedures` and `Measurement` domains.

## Predicted Entities

`SNOMED` codes from `Procedures` and `Measurements`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_snomed_procedures_measurements_en_3.3.2_3.0_1636985738813.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_snomed_procedures_measurements_en_3.3.2_3.0_1636985738813.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sent_biobert_clinical_base_cased", "en")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings") 

resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_clinical_snomed_procedures_measurements", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("cpt_code")

pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
resolver])

l_model = LightPipeline(pipelineModel)
result = l_model.fullAnnotate(['coronary calcium score', 'heart surgery', 'ct scan', 'bp value'])

```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sent_biobert_clinical_base_cased", "en")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_clinical_snomed_procedures_measurements", "en", "clinical/models) 
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("cpt_code")

val pipelineModel= new PipelineModel().setStages(Array(document_assembler, sbert_embedder, resolver))
val l_model = LightPipeline(pipelineModel)
val result = l_model.fullAnnotate(Array("coronary calcium score", "heart surgery", "ct scan", "bp value"))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.clinical_snomed_procedures_measurements").predict("""coronary calcium score""")
```

</div>

## Results

```bash
|    | chunk                  |      code | code_description              | all_k_code_desc                                                                 | all_k_codes                                                                                                                                                     |
|---:|:-----------------------|----------:|:------------------------------|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | coronary calcium score | 450360000 | Coronary artery calcium score | ['450360000', '450734004', '1086491000000104', '1086481000000101', '762241007'] | ['Coronary artery calcium score', 'Coronary artery calcium score', 'Dundee Coronary Risk Disk score', 'Dundee Coronary Risk rank', 'Dundee Coronary Risk Disk'] |
|  1 | heart surgery          |   2598006 | Open heart surgery            | ['2598006', '64915003', '119766003', '34068001', '233004008']                   | ['Open heart surgery', 'Operation on heart', 'Heart reconstruction', 'Heart valve replacement', 'Coronary sinus operation']                                     |
|  2 | ct scan                | 303653007 | CT of head                    | ['303653007', '431864000', '363023007', '418272005', '241577003']               | ['CT of head', 'CT guided injection', 'CT of site', 'CT angiography', 'CT of spine']                                                                            |
|  3 | bp value               |  75367002 | Blood pressure                | ['75367002', '6797001', '723232008', '46973005', '427732000']                   | ['Blood pressure', 'Mean blood pressure', 'Average blood pressure', 'Blood pressure taking', 'Speed of blood pressure response']                                |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_clinical_snomed_procedures_measurements|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_chunk_embeddings]|
|Output Labels:|[output]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on `SNOMED` code dataset with `sent_biobert_clinical_base_cased` sentence embeddings.
