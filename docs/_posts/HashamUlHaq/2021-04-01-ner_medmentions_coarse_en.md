---
layout: model
title: Detect mentions of general medical terms (coarse)
author: John Snow Labs
name: ner_medmentions_coarse
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract general medical terms in text like body parts, cells, genes, symptoms, etc in text using pretrained NER model.

## Predicted Entities

`Qualitative_Concept`, `Organization`, `Manufactured_Object`, `Amino_Acid,_Peptide,_or_Protein`, `Pharmacologic_Substance`, `Professional_or_Occupational_Group`, `Cell_Component`, `Neoplastic_Process`, `Substance`, `Laboratory_Procedure`, `Nucleic_Acid,_Nucleoside,_or_Nucleotide`, `Research_Activity`, `Gene_or_Genome`, `Indicator,_Reagent,_or_Diagnostic_Aid`, `Biologic_Function`, `Chemical`, `Mammal`, `Molecular_Function`, `Quantitative_Concept`, `Prokaryote`, `Mental_or_Behavioral_Dysfunction`, `Injury_or_Poisoning`, `Body_Location_or_Region`, `Spatial_Concept`, `Nucleotide_Sequence`, `Tissue`, `Pathologic_Function`, `Body_Substance`, `Fungus`, `Mental_Process`, `Medical_Device`, `Plant`, `Health_Care_Activity`, `Clinical_Attribute`, `Genetic_Function`, `Food`, `Therapeutic_or_Preventive_Procedure`, `Body_Part,_Organ,_or_Organ_Component`, `Geographic_Area`, `Virus`, `Biomedical_or_Dental_Material`, `Diagnostic_Procedure`, `Eukaryote`, `Anatomical_Structure`, `Organism_Attribute`, `Molecular_Biology_Research_Technique`, `Organic_Chemical`, `Cell`, `Daily_or_Recreational_Activity`, `Population_Group`, `Disease_or_Syndrome`, `Group`, `Sign_or_Symptom`, `Body_System`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_medmentions_coarse_en_3.0.0_3.0_1617260791003.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_medmentions_coarse", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_medmentions_coarse", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.medmentions").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_medmentions_coarse|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|