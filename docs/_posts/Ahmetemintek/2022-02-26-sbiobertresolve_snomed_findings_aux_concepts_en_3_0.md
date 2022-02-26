---
layout: model
title: Sentence Entity Resolver for SNOMED Concepts
author: John Snow Labs
name: sbiobertresolve_snomed_findings_aux_concepts
date: 2022-02-26
tags: [snomed, licensed, en, clinical, aux, ct]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.1.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to Snomed codes using sbiobert_base_cased_mli Sentence Bert Embeddings. This is also capable of extracting `Morph Abnormality`, `Procedure`, `Substance`, `Physical Object`, and `Body Structure` concepts of Snomed codes.

In the metadata, the `all_k_aux_labels` can be divided to get further information: `ground truth`, `concept`, and `aux`. For example, in the example shared below the ground truth is `Atherosclerosis`, concept is `Observation`, and aux is `Morph Abnormality`

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_aux_concepts_en_3.1.2_3.0_1645879611162.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical','en', 'clinical/models')\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ner_clinical = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("jsl_ner")

ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "jsl_ner"]) \
      .setOutputCol("ner_chunk")\

chunk2doc = Chunk2Doc() \
      .setInputCols("ner_chunk") \
      .setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")\
     .setDistanceFunction("EUCLIDEAN")

nlpPipeline= Pipeline(stages=[
                              documentAssembler,
                              sentenceDetector,
                              tokenizer,
                              word_embeddings,
                              ner_clinical,
                              ner_converter,
                              chunk2doc,
                              sbert_embedder,
                              snomed_resolver
])

text= """FINDINGS: The patient was found upon excision of the cyst that it contained a large Prolene suture, which is multiply knotted as it always is; beneath this was a very small incisional hernia, the hernia cavity, which contained omentum; the hernia was easily repaired"""

df= spark.createDataFrame([[text]]).toDF("text")

result= nlpPipeline.fit(df).transform(df)
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
      .setInputCols("document")
      .setOutputCol("sentence")

val tokenizer = Tokenizer() 
      .setInputCols("document") 
      .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical','en', 'clinical/models')
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")

val ner_clinical = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") 
      .setInputCols(Array("sentence", "token", "embeddings")) 
      .setOutputCol("jsl_ner")

val ner_converter = NerConverter() 
      .setInputCols(Array("sentence", "token", "jsl_ner")) 
      .setOutputCol("ner_chunk")

val chunk2doc = Chunk2Doc() 
      .setInputCols("ner_chunk") 
      .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')
      .setInputCols("ner_chunk_doc")
      .setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models")
     .setInputCols(Array("ner_chunk", "sbert_embeddings"))
     .setOutputCol("snomed_code")

val new nlpPipeine().setStages(Array(documentAssembler,
                                    sentenceDetector,
                                    tokenizer,
                                    word_embeddings,
                                    ner_clinical,
                                    ner_converter,
                                    chunk2doc,
                                    sbert_embedder,
                                    snomed_resolver))

val text= """FINDINGS: The patient was found upon excision of the cyst that it contained a large Prolene suture, which is multiply knotted as it always is; beneath this was a very small incisional hernia, the hernia cavity, which contained omentum; the hernia was easily repaired"""

val df = Seq(text).toDF(“text”) 

val result= nlpPipeline.fit(df).transform(df)
```
</div>

## Results

```bash
|    |   sent_id | ner_chunk                      | entity    |       snomed_code | all_codes                                                                                                                                                                                                                                                                                                                                                                                                                              | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|---:|----------:|:-------------------------------|:----------|------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |         1 | excision                       | TREATMENT |         180397004 | ['180397004', '65801008', '129304002', '257819000', '82868003', '119858002', '20418004', '246386008', '120086003', '701710001', '265901005', '440258006', '150339009', '246159003', '787139004', '26418007', '119625009', '119647009', '277261002']                                                                                                                                                                                    | excision from organ noc: [sinus tract] or [fistula]:::excision:::excision - action:::surgical excision:::margins of excision:::mouth excision:::v excision:::method of excision:::eye excision:::subcutaneous-catheter tunneler:::excision of skin:::excision of skin:::excision of skin:::extent of excision:::piecemeal excision (procedure):::dorsolumbar fusion:::arm excision (procedure):::hand excision:::excision biopsy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|  1 |         1 | the cyst                       | PROBLEM   |         246178003 | ['246178003', '103552005', '441457006', '264515009', '367643001', '258420003', '734100004', '119368000', '447030009', '110408007', '734110008', '419093005', '87373006', '254677004', '7102003', '42323001', '79134002', '195497001', '103616003', '20476009', '20462008', '254688005', '19633006']                                                                                                                                    | form of cyst:::cyst:::cyst:::cyst:::cyst:::cyst tissue:::well-differentiated papillary mesothelioma:::specimen from cyst:::ciliated cyst:::congenital vascular anomaly, macular type:::cyst fluid (substance):::pilar cyst:::pilar cyst:::pilar cyst:::omental cyst:::eruption cyst:::chylous cyst:::chylous cyst:::multilocular cyst:::cystic:::ameloblastoma:::glandular cyst:::mucinous cyst                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|  2 |         1 | a large Prolene suture         | TREATMENT | 20594411000001105 | ['20594411000001105', '7267511000001100', '20125511000001105', '463182000', '20085011000001103', '19823811000001102', '15097411000001106', '463858007', '257395002', '401800008', '14585811000001105', '34238511000001104', '14585511000001107', '641311000001106', '257394003', '20085111000001102', '401801007', '15038611000001104', '15041911000001100', '939611000001103', '37189611000001108', '464012004', '15040111000001109'] | finger stalls plastic medium:::portia disposable gloves polythene medium (bray group ltd):::silk mittens 8-14 years:::polybutester suture:::skinnies silk gloves large child blue (dermacea ltd):::soft silicone wound contact dressing sterile 8cm x 10cm 5 dressing:::non-absorbable synthetic polypropylene monofilament suture 1.5gauge 45cm length with 24mm 3/8 curved reverse cutting needle:::orthodontic appliance automated cleaner:::polyester suture:::suspensory bandage cotton type 1 extra large (physical object):::premier nitrile lilac gloves large (shermond):::carmoisine:::premier nitrile lilac gloves medium (shermond):::cotton stockinette bleached heavyweight 7.5cm (e sallis ltd):::prolene suture:::skinnies silk gloves large child blue (dermacea ltd) 2 device:::suspensory bandage cotton type 1 large (physical object):::skinnies viscose stockinette vest long sleeve large adult blue (dermacea ltd):::skinnies viscose stockinette gloves large child blue (dermacea ltd):::cotton stockinette bleached heavyweight 5cm (e sallis ltd):::skinnies am child dermasock large pink (dermacea ltd):::poliglecaprone suture:::skinnies viscose stockinette vest long sleeve large adult grey (dermacea ltd) 1 device |
|  3 |         1 | multiply knotted               | TREATMENT |         705645008 | ['705645008', '251246008', '251252009', '247927001', '86586007', '247557001', '371922008', '251202000', '705651003', '285345008', '58083006', '247554008', '247929003', '251208001', '249899003', '205596006', '69488000', '276482004', '161873000', '139130002', '257947000', '247558006']                                                                                                                                            | suturing instrument:::ecg complex shape:::qrs complex shape (observable entity):::repetitive complex twisting movements:::bowstringing:::spangled hair:::multiple irregularities:::ecg complex characteristics:::multifilament suture (physical object):::multiple lacerations:::multiloculated:::kinked hairs:::repetitive complex twisting movements of limbs:::qrs complex duration (observable entity):::large movements of limb:::beaded hair:::beaded hair:::beaded hair:::heavy legs:::heavy legs:::multiple kirschner wiring:::spiralled hair                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|  4 |         1 | a very small incisional hernia | PROBLEM   |         155752004 | ['155752004', '196894007', '266513000', '415772007', '266514006', '155753009', '196867008', '155754003', '266515007', '767675006', '125223003', '266441003', '196816005', '196924004', '773271009', '266443000', '196895008']                                                                                                                                                                                                          | simple incisional hernia:::simple incisional hernia:::simple incisional hernia:::uncomplicated ventral incisional hernia:::umbilical hernia - simple:::umbilical hernia - simple:::simple umbilical hernia:::paraumbilical hernia - simple:::paraumbilical hernia - simple:::simple left inguinal hernia:::congenital imperforation, high:::unilateral simple inguinal hernia (situation):::unilateral simple inguinal hernia:::simple perineal hernia:::simple perineal hernia:::simple epigastric hernia:::simple epigastric hernia                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|  5 |         1 | the hernia cavity              | PROBLEM   |         112639008 | ['112639008', '52515009', '359801000', '414403008', '147780008', '155737006', '196799009', '443827002', '30477007', '236042008', '128545000', '236043003', '140528004', '125256002', '110414000', '196876001']                                                                                                                                                                                                                         | protrusion:::hernia:::hernia:::hernia:::notification of whooping cough:::hernia of abdominal cavity:::hernia of abdominal cavity:::hernia of body cavity structure:::internal hernia:::obstructed internal hernia:::hernia of abdominal wall:::strangulated internal hernia:::o/e - irreducible hernia (finding):::lateral protrusion:::odontogenic cyst with ameloblastomatous change:::incisional hernia                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|  6 |         1 | the hernia                     | PROBLEM   |          52515009 | ['52515009', '359801000', '414403008', '147780008', '112639008', '83836003', '125256002', '14778003', '236003', '196967007', '236020007', '385525003', '264544006', '88078006', '276915000', '50063009', '27138711000001101', '40421008', '110418002', '140528004', '30477007']                                                                                                                                                        | hernia:::hernia:::hernia:::notification of whooping cough:::protrusion:::omental hernia:::lateral protrusion:::herniation:::incision of vein:::richter's hernia:::richter's hernia:::littre's hernia:::littre's hernia:::intermuscular hernia:::hernia - lesion:::crural hernia:::hydrocolloid dressing thin semi-permeable sterile without adhesive border 5cm x 5cm square:::vesical hernia:::irreducible hernia:::o/e - irreducible hernia (finding):::internal hernia                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|  7 |         1 | repaired                       | TREATMENT |          50826004 | ['50826004', '4365001', '257903006', '33714007', '260938008', '398004007', '35174006', '18867006', '1241001', '46998006', '260729003', '782902008', '71861002', '73504009', '90910008', '1727009', '17275008', '260373001', '723506003', '308286001', '15635006', '371152001', '255396000']                                                                                                                                            | repaired:::repair:::repair:::corrected:::restoration:::relieved:::reversible:::healing:::relieved by:::marked:::placement:::placement:::placement:::classified:::healed:::deoxylimonate a-ring-lactonase:::compensated:::detected:::resolved:::repairer:::re:::assisted:::acquired                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_findings_aux_concepts|
|Compatibility:|Spark NLP for Healthcare 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Size:|4.7 GB|
|Case sensitive:|false|