---
layout: model
title: do_NOT_MERGE_NLP_LAB_TEST
author: John Snow Labs
name: ner_Breast_Cancer_Training_QA
date: 2023-12-18
tags: [en, open_source, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.1.2
spark_version: 3.2
supported: true
engine: tensorflow
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

esdfsdf

## Predicted Entities

`Past`, `Dosage`, `Oncogene`, `Grade`, `CancerDx`, `Pathology_Result`, `Female_Reproductive_Status`, `Lymph_Node_Modifier`, `ResponseToTreatment`, `RadiationDose`, `Death_Entity`, `Frequency`, `Tumor_Finding`, `HistologicalType`, `RelativeDate`, `DocSplit_NOT_NER`, `Invasion`, `SiteLung`, `SiteBreast`, `SiteBone`, `Absent`, `Adenopathy`, `SiteLiver`, `Cycleday`, `SizeTrend`, `UnspecificTherapy`, `CancerSurgery`, `Duration`, `Planned`, `CancerScore`, `PathologyTest`, `Cyclenumber`, `ImagingTest`, `TargetedTherapy`, `Gender`, `Age`, `BenignTumor`, `Biomarker_Result`, `Chemotherapy`, `Direction`, `LineOfTherapy`, `Race_Ethnicity`, `HormonalTherapy`, `Hypothetical`, `CancerScoreValue`, `Staging`, `Cyclelength`, `TumorSize`, `SiteBrain`, `Date`, `SiteOtherBodyPart`, `Immunotherapy`, `Possible`, `Family`, `Form`, `Cycledose`, `SiteLymphNode`, `Route`, `Metastasis`, `Cyclecount`, `Suspected`, `Confirmed`, `Biomarker`, `PalliativeTreatment`, `PerformanceStatus`, `Radiotherapy`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_Breast_Cancer_Training_QA_en_5.1.2_3.2_1702898918791.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_Breast_Cancer_Training_QA_en_5.1.2_3.2_1702898918791.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()
			.setInputCol("text")
			.setOutputCol("document")

sentence_detector = SentenceDetector()
			.setInputCols(["document"])
			.setOutputCol("sentence")
			.setCustomBounds([""])

tokenizer = Tokenizer()
		.setInputCols(["sentence"])
		.setOutputCol(\"token\")
		.setSplitChars(['-'])"

word_embeddings = WordEmbeddingsModel()
			.pretrained("glove_100d", "en" )
			.setInputCols(["sentence", "token"])
			.setOutputCol("embeddings")

ner = NerDLModel().pretrained("ner_Breast_Cancer_Training_QA", "en" )
		.setInputCols(["sentence", "token", "embeddings"])
		.setOutputCol("ner")

ner_converter = NerConverter()
			.setInputCols(["sentence", "token", "ner"])
			.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[document_assembler,
			    sentence_detector,
			    tokenizer,
			    word_embeddings,
			    ner,
			    ner_converter])

data = spark.createDataFrame([["SAMPLE_TEXT"]]).toDF("text")
result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
TEST 
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_Breast_Cancer_Training_QA|
|Type:|ner|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|3.0 MB|
|Dependencies:|glove_100d|