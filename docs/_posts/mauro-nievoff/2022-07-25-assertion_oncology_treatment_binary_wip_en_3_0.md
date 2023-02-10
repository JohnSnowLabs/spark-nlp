---
layout: model
title: Detect Assertion Status for Oncology Treatments
author: John Snow Labs
name: assertion_oncology_treatment_binary_wip
date: 2022-07-25
tags: [licensed, english, clinical, assertion, oncology, cancer, treatment, en]
task: Assertion Status
language: en
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
published: false
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Assign assertion status to oncology treatment entities extracted by ner_oncology_wip based on their context. This model predicts if a treatment mentioned in text has been used by the patient (in the past or in the present) or not used (mentioned as absent, as treatment plan or as something hypothetical).

## Predicted Entities

`Present_Or_Past`, `Hypothetical_Or_Absent`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_oncology_treatment_binary_wip_en_3.5.0_3.0_1658774066204.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_oncology_treatment_binary_wip_en_3.5.0_3.0_1658774066204.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(['document'])\
    .setOutputCol('sentence')

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained('embeddings_clinical', 'en', 'clinical/models')\
    .setInputCols(["sentence", 'token']) \
    .setOutputCol("embeddings")

ner_oncology = MedicalNerModel.pretrained('ner_oncology_wip', 'en', 'clinical/models')\
                    .setInputCols(["sentence", "token", "embeddings"])\
                    .setOutputCol("ner")

ner_oncology_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setWhiteList(['Chemotherapy', 'Immunotherapy', 'Hormonal_Therapy', 'Targeted_Therapy', 'Unspecific_Therapy', 'Cancer_Therapy', 'Radiotherapy'])

assertion = AssertionDLModel.pretrained("assertion_oncology_treatment_binary_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
 
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner_oncology,
                            ner_oncology_converter,
			    assertion])

data = spark.createDataFrame([["The patient underwent a mastectomy three years ago. She continued with paclitaxel and trastuzumab for her breast cancer. She was not treated with radiotherapy. We discussed the possibility of using chemotherapy."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
		.setInputCols(Array("document"))
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols(Array("sentence"))
		.setOutputCol("token")
	
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
		.setInputCols(Array("sentence", "token"))
	    .setOutputCol("embeddings")
  
val ner_oncology = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models")
		.setInputCols(Array("sentence", "token", "embeddings"))
		.setOutputCol("ner")

val ner_oncology_converter = new NerConverter()
		.setInputCols(Array("sentence", "token", "ner"))
		.setOutputCol("ner_chunk")
		.setWhiteList(Array('Chemotherapy', 'Immunotherapy', 'Hormonal_Therapy', 'Targeted_Therapy', 'Unspecific_Therapy', 'Cancer_Therapy', 'Radiotherapy'))

val assertion = AssertionDLModel.pretrained("assertion_oncology_treatment_binary_wip", "en", "clinical/models")
		.setInputCols(Array("sentence", "ner_chunk", "embeddings"))
		.setOutputCol("assertion")
 
val pipeline = new Pipeline().setStages(Array(
					documentAssembler, 
					sentenceDetector, 
					tokenizer, 
					embeddings, 
					ner_oncology, 
					ner_oncology_converter,
					assertion))


val data = Seq("""The patient underwent a mastectomy three years ago. She continued with paclitaxel and trastuzumab for her breast cancer. She was not treated with radiotherapy. We discussed the possibility of using chemotherapy.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.assert.oncology_treatment_binary_wip").predict("""The patient underwent a mastectomy three years ago. She continued with paclitaxel and trastuzumab for her breast cancer. She was not treated with radiotherapy. We discussed the possibility of using chemotherapy.""")
```
</div>

## Results

```bash
+------------+-----+---+----------------+----------------------+
|       chunk|begin|end|       ner_label|      assertion_status|
+------------+-----+---+----------------+----------------------+
|  mastectomy|   24| 33|  Cancer_Surgery|       Present_Or_Past|
|  paclitaxel|   71| 80|    Chemotherapy|       Present_Or_Past|
| trastuzumab|   86| 96|Targeted_Therapy|       Present_Or_Past|
|radiotherapy|  146|157|    Radiotherapy|Hypothetical_Or_Absent|
|chemotherapy|  198|209|    Chemotherapy|Hypothetical_Or_Absent|
+------------+-----+---+----------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_oncology_treatment_binary_wip|
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion_pred]|
|Language:|en|
|Size:|1.4 MB|

## References

Trained on case reports sampled from PubMed, and annotated in-house.

## Benchmarking

```bash
label	 		tp	fp	fn	prec	 	rec	 	f1
Present_Or_Past	 	50	13	14	0.7936508	0.78125	 	0.78740156
Hypothetical_Or_Absent	68	14	13	0.8292683	0.83950615	0.83435583
Macro-average		118 	27	27	0.8114595	0.8103781	0.8109184
Micro-average	 	118 	27	27	0.8137931 	0.8137931 	0.81379306
```
