---
layout: model
title: Detect Oncology Entities
author: John Snow Labs
name: ner_oncology_wip
date: 2022-07-25
tags: [licensed, english, clinical, ner, oncology, cancer, biomarker, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
published: false
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained named entity recognition (NER) model is a deep learning model for detecting entities related to cancer diagnosis (such as staging, grade or mentiones of metastasis), cancer treatments (chemotherapy, targeted therapy or surgical procedures, among other), and oncological tests (pathology tests, biomarkers, oncogenes, etc). The model was trained using `MedicalNerApproach` annotator that allows to train generic NER models based on Neural Networks.

## Predicted Entities

`Gender`, `Age`, `Race_Ethnicity`, `Date`, `Cancer_Dx`, `Metastasis`, `Invasion`, `Histological_Type`, `Grade`, `Tumor_Finding`, `Staging`, `Tumor_Size`, `Oncogene`, `Biomarker`, `Biomarker_Result`, `Performance_Status`, `Pathology_Test`, `Pathology_Result`, `Smoking_Status`, `Anatomical_Site`, `Direction`, `Site_Lymph_Node`, `Chemotherapy`, `Immunotherapy`, `Targeted_Therapy`, `Hormonal_Therapy`, `Unspecific_Therapy`, `Radiotherapy`, `Cancer_Surgery`, `Line_Of_Therapy`, `Response_To_Treatment`, `Radiation_Dose`, `Duration`, `Frequency`, `Cycle_Number`, `Cycle_Day`, `Dosage`, `Route`, `Relative_Date`, `Imaging_Test`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_wip_en_3.5.0_3.0_1658771306053.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner_oncology,
                            ner_oncology_converter])

data = spark.createDataFrame([["She then sought medical attention for a breast lump that she had noticed for the past few months. This was clinically diagnosed as breast cancer. She subsequently underwent right wide local excision of the mass and axillary clearance. Histology revealed 28mm grade 3 oestrogen receptor positive, human epidermal growth factor receptor 2 negative ductal carcinoma involving 12 of 14 axillary nodes. An oncology referral was made."]]).toDF("text")

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
 
val pipeline = new Pipeline().setStages(Array(
					documentAssembler, 
					sentenceDetector, 
					tokenizer, 
					embeddings, 
					ner_oncology, 
					ner_oncology_converter))


val data = Seq("""She then sought medical attention for a breast lump that she had noticed for the past few months. This was clinically diagnosed as breast cancer. She subsequently underwent right wide local excision of the mass and axillary clearance. Histology revealed 28mm grade 3 oestrogen receptor positive, human epidermal growth factor receptor 2 negative ductal carcinoma involving 12 of 14 axillary nodes. An oncology referral was made.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.oncology_wip").predict("""She then sought medical attention for a breast lump that she had noticed for the past few months. This was clinically diagnosed as breast cancer. She subsequently underwent right wide local excision of the mass and axillary clearance. Histology revealed 28mm grade 3 oestrogen receptor positive, human epidermal growth factor receptor 2 negative ductal carcinoma involving 12 of 14 axillary nodes. An oncology referral was made.""")
```
</div>

## Results

```bash
+----------------------------------------+-----+---+-----------------+
|                                   chunk|begin|end|        ner_label|
+----------------------------------------+-----+---+-----------------+
|                                     She|    0|  2|           Gender|
|                                  breast|   40| 45|  Anatomical_Site|
|                                    lump|   47| 50|    Tumor_Finding|
|                                     she|   57| 59|           Gender|
|                 for the past few months|   73| 95|         Duration|
|                           breast cancer|  131|143|        Cancer_Dx|
|                                     She|  146|148|           Gender|
|                                   right|  173|177|        Direction|
|                     wide local excision|  179|197|   Cancer_Surgery|
|                                    mass|  206|209|    Tumor_Finding|
|                      axillary clearance|  215|232|  Anatomical_Site|
|                               Histology|  235|243|   Pathology_Test|
|                      oestrogen receptor|  267|284|        Biomarker|
|                                positive|  286|293| Biomarker_Result|
|human epidermal growth factor receptor 2|  296|335|         Oncogene|
|                                negative|  337|344| Biomarker_Result|
|                                  ductal|  346|351|Histological_Type|
|                  carcinoma involving 12|  353|374|        Cancer_Dx|
|                       14 axillary nodes|  379|395|  Site_Lymph_Node|
+----------------------------------------+-----+---+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_wip|
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|965.8 KB|

## References

Trained on case reports sampled from PubMed, and annotated in-house.

## Benchmarking

```bash
label                   tp	fp	fn	total	precision	recall		f1
Histological_Type	181.0	48.0	48.0	229.0	0.790393	0.790393	0.790393
Anatomical_Site		1166.0	401.0	208.0	1374.0	0.744097	0.848617	0.792928
Direction		391.0	75.0	159.0	550.0	0.839056	0.710909	0.769685
Staging			187.0	28.0	32.0	219.0	0.869767	0.853881	0.861751
Imaging_Test		959.0	119.0	120.0	1079.0	0.889610	0.888786	0.889198
Cycle_Number		162.0	43.0	25.0	187.0	0.790244	0.866310	0.826531
Tumor_Finding		446.0	62.0	96.0	542.0	0.877953	0.822878	0.849524
Site_Lymph_Node		414.0	86.0	73.0	487.0	0.828000	0.850103	0.838906
Invasion		21.0	8.0	28.0	49.0	0.724138	0.428571	0.538462
Response_To_Treatment	249.0	64.0	112.0	361.0	0.795527	0.689751	0.738872
Smoking_Status		18.0	10.0	7.0	25.0	0.642857	0.720000	0.679245
Tumor_Size		623.0	69.0	37.0	660.0	0.900289	0.943939	0.921598
Age			416.0	1.0	23.0	439.0	0.997602	0.947608	0.971963
Biomarker_Result	539.0	122.0	202.0	741.0	0.815431	0.727395	0.768902
Unspecific_Therapy	57.0	30.0	52.0	109.0	0.655172	0.522936	0.581633
Chemotherapy		415.0	29.0	25.0	440.0	0.934685	0.943182	0.938914
Targeted_Therapy	104.0	33.0	10.0	114.0	0.759124	0.912281	0.828685
Radiotherapy		104.0	10.0	26.0	130.0	0.912281	0.800000	0.852459
Performance_Status	61.0	6.0	35.0	96.0	0.910448	0.635417	0.748466
Pathology_Test		352.0	68.0	118.0	470.0	0.838095	0.748936	0.791011
Cancer_Surgery		291.0	41.0	38.0	329.0	0.876506	0.884498	0.880484
Line_Of_Therapy		64.0	5.0	11.0	75.0	0.927536	0.853333	0.888889
Pathology_Result	154.0	156.0	82.0	236.0	0.496774	0.652542	0.564103
Hormonal_Therapy	62.0	3.0	14.0	76.0	0.953846	0.815789	0.879433
Biomarker		734.0	236.0	103.0	837.0	0.756701	0.876941	0.812396
Immunotherapy		24.0	7.0	19.0	43.0	0.774194	0.558140	0.648649
Cycle_Day		58.0	21.0	10.0	68.0	0.734177	0.852941	0.789116
Frequency		192.0	17.0	53.0	245.0	0.918660	0.783673	0.845815
Route			26.0	3.0	41.0	67.0	0.896552	0.388060	0.541667
Duration		230.0	88.0	67.0	297.0	0.723270	0.774411	0.747967
Metastasis		171.0	2.0	22.0	193.0	0.988439	0.886010	0.934426
Cancer_Dx		568.0	59.0	59.0	627.0	0.905901	0.905901	0.905901
Grade			15.0	9.0	73.0	88.0	0.625000	0.170455	0.267857
Date			385.0	21.0	12.0	397.0	0.948276	0.969773	0.958904
Relative_Date		348.0	85.0	91.0	439.0	0.803695	0.792711	0.798165
Race_Ethnicity		21.0	1.0	3.0	24.0	0.954545	0.875000	0.913043
Gender			619.0	12.0	10.0	629.0	0.980983	0.984102	0.982540
Dosage			512.0	45.0	107.0	619.0	0.919210	0.827141	0.870748
Oncogene		283.0	23.0	120.0	403.0	0.924837	0.702233	0.798307
Radiation_Dose		53.0	13.0	3.0	56.0	0.803030	0.946429	0.868852
Macro-average 		-	-	-	-	-		-		0.796909
Micro-average 		-	-	-	-	-		-		0.835757
```
