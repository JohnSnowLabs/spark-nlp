{%- capture title -%}
ZeroShotRelationExtractionModel
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

`ZeroShotRelationExtractionModel` implements zero-shot binary relations extraction by utilizing `BERT` transformer models trained on the NLI (Natural Language Inference) task. 

The model inputs consists of documents/sentences and paired NER chunks, usually obtained by `RENerChunksFilter`. The definitions of relations which are extracted is given by a dictionary structures, specifying a set of statements regarding the relationship of named entities. 

These statements are automatically appended to each document in the dataset and the NLI model is used to determine whether a particular relationship between entities.

For available pretrained models please see the [NLP Models Hub](https://nlp.johnsnowlabs.com/models?task=Zero-Shot-Classification Models Hub).

{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, DOCUMENT 
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_python_medical -%}

documenter = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentencer = (
    SentenceDetectorDLModel.pretrained(
        "sentence_detector_dl_healthcare", "en", "clinical/models"
    )
    .setInputCols(["document"])
    .setOutputCol("sentences")
)

tokenizer = Tokenizer().setInputCols(["sentences"]).setOutputCol("tokens")

words_embedder = (
    WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(["sentences", "tokens"])
    .setOutputCol("embeddings")
)

ner_clinical = (
    MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(["sentences", "tokens", "embeddings"])
    .setOutputCol("ner_clinical")
)

ner_clinical_converter = (
    NerConverter()
    .setInputCols(["sentences", "tokens", "ner_clinical"])
    .setOutputCol("ner_clinical_chunks")
    .setWhiteList(["PROBLEM", "TEST"])
)  # PROBLEM-TEST-TREATMENT

ner_posology = (
    MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")
    .setInputCols(["sentences", "tokens", "embeddings"])
    .setOutputCol("ner_posology")
)

ner_posology_converter = (
    NerConverter()
    .setInputCols(["sentences", "tokens", "ner_posology"])
    .setOutputCol("ner_posology_chunks")
    .setWhiteList(["DRUG"])
)  # DRUG-FREQUENCY-DOSAGE-DURATION-FORM-ROUTE-STRENGTH

chunk_merger = (
    ChunkMergeApproach()
    .setInputCols("ner_clinical_chunks", "ner_posology_chunks")
    .setOutputCol("merged_ner_chunks")
)


## ZERO-SHOT RE Starting...

pos_tagger = (
    PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(["sentences", "tokens"])
    .setOutputCol("pos_tags")
)

dependency_parser = (
    DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(["document", "pos_tags", "tokens"])
    .setOutputCol("dependencies")
)

re_ner_chunk_filter = (
    RENerChunksFilter()
    .setRelationPairs(["problem-test", "problem-drug"])
    .setMaxSyntacticDistance(4)
    .setDocLevelRelations(False)
    .setInputCols(["merged_ner_chunks", "dependencies"])
    .setOutputCol("re_ner_chunks")
)

re_model = (
    ZeroShotRelationExtractionModel.pretrained(
        "re_zeroshot_biobert", "en", "clinical/models"
    )
    .setInputCols(["re_ner_chunks", "sentences"])
    .setOutputCol("relations")
    .setRelationalCategories(
        {
            "ADE": ["{DRUG} causes {PROBLEM}."],
            "IMPROVE": ["{DRUG} improves {PROBLEM}.", "{DRUG} cures {PROBLEM}."],
            "REVEAL": ["{TEST} reveals {PROBLEM}."],
        }
    )
    .setMultiLabel(True)
)

pipeline = sparknlp.base.Pipeline().setStages(
    [
        documenter,
        sentencer,
        tokenizer,
        words_embedder,
        ner_clinical,
        ner_clinical_converter,
        ner_posology,
        ner_posology_converter,
        chunk_merger,
        pos_tagger,
        dependency_parser,
        re_ner_chunk_filter,
        re_model,
    ]
)

sample_text = "Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer."

data = spark.createDataFrame([[sample_text]]).toDF("text")

model = pipeline.fit(data)
results = model.transform(data)

from pyspark.sql import functions as F

results.select(
    F.explode(F.arrays_zip(results.relations.metadata, results.relations.result)).alias(
        "cols"
    )
).select(
    F.expr("cols['0']['sentence']").alias("sentence"),
    F.expr("cols['0']['entity1_begin']").alias("entity1_begin"),
    F.expr("cols['0']['entity1_end']").alias("entity1_end"),
    F.expr("cols['0']['chunk1']").alias("chunk1"),
    F.expr("cols['0']['entity1']").alias("entity1"),
    F.expr("cols['0']['entity2_begin']").alias("entity2_begin"),
    F.expr("cols['0']['entity2_end']").alias("entity2_end"),
    F.expr("cols['0']['chunk2']").alias("chunk2"),
    F.expr("cols['0']['entity2']").alias("entity2"),
    F.expr("cols['0']['hypothesis']").alias("hypothesis"),
    F.expr("cols['0']['nli_prediction']").alias("nli_prediction"),
    F.expr("cols['1']").alias("relation"),
    F.expr("cols['0']['confidence']").alias("confidence"),
).show(
    truncate=70
)
+--------+-------------+-----------+-----------+-------+-------------+-----------+--------+-------+------------------------------+--------------+--------+----------+
sentence|entity1_begin|entity1_end|     chunk1|entity1|entity2_begin|entity2_end|  chunk2|entity2|                    hypothesis|nli_prediction|relation|confidence|
+--------+-------------+-----------+-----------+-------+-------------+-----------+--------+-------+------------------------------+--------------+--------+----------+
       0|            0|         10|Paracetamol|   DRUG|           38|         45|sickness|PROBLEM|Paracetamol improves sickness.|        entail| IMPROVE|0.98819494|
       0|            0|         10|Paracetamol|   DRUG|           26|         33|headache|PROBLEM|Paracetamol improves headache.|        entail| IMPROVE| 0.9929625|
       1|           48|         58|An MRI test|   TEST|           80|         85|  cancer|PROBLEM|   An MRI test reveals cancer.|        entail|  REVEAL| 0.9760039|
+--------+-------------+-----------+-----------+-------+-------------+-----------+--------+-------+------------------------------+--------------+--------+----------+

{%- endcapture -%}

{%- capture model_python_finance -%}

document_assembler = (
    nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
)

sentence_detector = (
    nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
    .setInputCols(["document"])
    .setOutputCol("sentence")
)

tokenizer = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

embeddings = (
    nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en")
    .setInputCols(["sentence", "token"])
    .setOutputCol("embeddings")
)

ner_model = (
    finance.NerModel.pretrained("finner_financial_small", "en", "finance/models")
    .setInputCols(["sentence", "token", "embeddings"])
    .setOutputCol("ner")
)
ner_converter = (
    nlp.NerConverter()
    .setInputCols(["sentence", "token", "ner"])
    .setOutputCol("ner_chunk")
)

re_model = (
    finance.ZeroShotRelationExtractionModel.pretrained(
        "finre_zero_shot", "en", "finance/models"
    )
    .setInputCols(["ner_chunk", "sentence"])
    .setOutputCol("relations")
    .setMultiLabel(False)
)

re_model.setRelationalCategories(
    {
        "profit_decline_by": [
            "{PROFIT_DECLINE} decreased by {AMOUNT} from",
            "{PROFIT_DECLINE} decreased by {AMOUNT} to",
        ],
        "profit_decline_by_per": [
            "{PROFIT_DECLINE} decreased by a {PERCENTAGE} from",
            "{PROFIT_DECLINE} decreased by a {PERCENTAGE} to",
        ],
        "profit_decline_from": [
            "{PROFIT_DECLINE} decreased from {AMOUNT}",
            "{PROFIT_DECLINE} decreased from {AMOUNT} for the year",
        ],
        "profit_decline_from_per": [
            "{PROFIT_DECLINE} decreased from {PERCENTAGE} to",
            "{PROFIT_DECLINE} decreased from {PERCENTAGE} to a total of",
        ],
        "profit_decline_to": ["{PROFIT_DECLINE} to {AMOUNT}"],
        "profit_increase_from": ["{PROFIT_INCREASE} from {AMOUNT}"],
        "profit_increase_to": ["{PROFIT_INCREASE} to {AMOUNT}"],
        "expense_decrease_by": ["{EXPENSE_DECREASE} decreased by {AMOUNT}"],
        "expense_decrease_by_per": ["{EXPENSE_DECREASE} decreased by a {PERCENTAGE}"],
        "expense_decrease_from": ["{EXPENSE_DECREASE} decreased from {AMOUNT}"],
        "expense_decrease_to": [
            "{EXPENSE_DECREASE} for a total of {AMOUNT} for the fiscal year"
        ],
        "has_date": [
            "{AMOUNT} for the fiscal year ended {FISCAL_YEAR}",
            "{PERCENTAGE} for the fiscal year ended {FISCAL_YEAR}",
        ],
    }
)

pipeline = nlp.Pipeline(
    stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        re_model,
    ]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

light_model = nlp.LightPipeline(model)

sample_text = """License fees revenue decreased 40 %, or $ 0.5 million to $ 0.7 million for the year ended December 31, 2020 compared to $ 1.2 million for the year ended December 31, 2019. Services revenue increased 4 %, or $ 1.1 million, to $ 25.6 million for the year ended December 31, 2020 from $ 24.5 million for the year ended December 31, 2019. Costs of revenue, excluding depreciation and amortization increased by $ 0.1 million, or 2 %, to $ 8.8 million for the year ended December 31, 2020 from $ 8.7 million for the year ended December 31, 2019.  Also, a decrease in travel costs of $ 0.4 million due to travel restrictions caused by the global pandemic. As a percentage of revenue, cost of revenue, excluding depreciation and amortization was 34 % for each of the years ended December 31, 2020 and 2019. Sales and marketing expenses decreased 20 %, or $ 1.5 million, to $ 6.0 million for the year ended December 31, 2020 from $ 7.5 million for the year ended December 31, 2019"""

data = spark.createDataFrame([[sample_text]]).toDF("text")

result = model.transform(data)
result.selectExpr("explode(relations) as relation").show(truncate=False)
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|relation                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|{category, 8462, 8693, has_date, {entity1_begin -> 227, relation -> has_date, hypothesis -> 25.6 million for the fiscal year ended December 31, 2019, confidence -> 0.8744761, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 332, entity1_end -> 238, entity2_begin -> 316, entity2 -> FISCAL_YEAR, chunk1 -> 25.6 million, sentence -> 1}, []}                                          |
|{category, 4643, 4873, has_date, {entity1_begin -> 31, relation -> has_date, hypothesis -> 40 for the fiscal year ended December 31, 2019, confidence -> 0.7889031, nli_prediction -> entail, entity1 -> PERCENTAGE, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 169, entity1_end -> 32, entity2_begin -> 153, entity2 -> FISCAL_YEAR, chunk1 -> 40, sentence -> 0}, []}                                                            |
|{category, 13507, 13748, expense_decrease_from, {entity1_begin -> 799, relation -> expense_decrease_from, hypothesis -> Sales and marketing expenses decreased from 7.5 million, confidence -> 0.9770538, nli_prediction -> entail, entity1 -> EXPENSE_DECREASE, syntactic_distance -> undefined, chunk2 -> 7.5 million, entity2_end -> 933, entity1_end -> 826, entity2_begin -> 923, entity2 -> AMOUNT, chunk1 -> Sales and marketing expenses, sentence -> 5}, []}|
|{category, 5354, 5593, has_date, {entity1_begin -> 59, relation -> has_date, hypothesis -> 0.7 million for the fiscal year ended December 31, 2020, confidence -> 0.6718765, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 106, entity1_end -> 69, entity2_begin -> 90, entity2 -> FISCAL_YEAR, chunk1 -> 0.7 million, sentence -> 0}, []}                                               |
|{category, 6490, 6697, profit_increase_to, {entity1_begin -> 172, relation -> profit_increase_to, hypothesis -> Services revenue to 25.6 million, confidence -> 0.9674029, nli_prediction -> entail, entity1 -> PROFIT_INCREASE, syntactic_distance -> undefined, chunk2 -> 25.6 million, entity2_end -> 238, entity1_end -> 187, entity2_begin -> 227, entity2 -> AMOUNT, chunk1 -> Services revenue, sentence -> 1}, []}                                           |
|{category, 4412, 4642, has_date, {entity1_begin -> 31, relation -> has_date, hypothesis -> 40 for the fiscal year ended December 31, 2020, confidence -> 0.778003, nli_prediction -> entail, entity1 -> PERCENTAGE, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 106, entity1_end -> 32, entity2_begin -> 90, entity2 -> FISCAL_YEAR, chunk1 -> 40, sentence -> 0}, []}                                                              |
|{category, 13989, 14221, has_date, {entity1_begin -> 838, relation -> has_date, hypothesis -> 20 for the fiscal year ended December 31, 2020, confidence -> 0.8545547, nli_prediction -> entail, entity1 -> PERCENTAGE, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 914, entity1_end -> 839, entity2_begin -> 898, entity2 -> FISCAL_YEAR, chunk1 -> 20, sentence -> 5}, []}                                                        |
|{category, 11157, 11314, expense_decrease_by, {entity1_begin -> 561, relation -> expense_decrease_by, hypothesis -> travel costs decreased by 0.4 million, confidence -> 0.9946776, nli_prediction -> entail, entity1 -> EXPENSE_DECREASE, syntactic_distance -> undefined, chunk2 -> 0.4 million, entity2_end -> 589, entity1_end -> 572, entity2_begin -> 579, entity2 -> AMOUNT, chunk1 -> travel costs, sentence -> 3}, []}                                      |
|{category, 5114, 5353, has_date, {entity1_begin -> 42, relation -> has_date, hypothesis -> 0.5 million for the fiscal year ended December 31, 2019, confidence -> 0.77566886, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 169, entity1_end -> 52, entity2_begin -> 153, entity2 -> FISCAL_YEAR, chunk1 -> 0.5 million, sentence -> 0}, []}                                             |
|{category, 6281, 6489, profit_increase_from, {entity1_begin -> 172, relation -> profit_increase_from, hypothesis -> Services revenue from 1.1 million, confidence -> 0.96610945, nli_prediction -> entail, entity1 -> PROFIT_INCREASE, syntactic_distance -> undefined, chunk2 -> 1.1 million, entity2_end -> 219, entity1_end -> 187, entity2_begin -> 209, entity2 -> AMOUNT, chunk1 -> Services revenue, sentence -> 1}, []}                                      |
|{category, 9199, 9471, has_date, {entity1_begin -> 408, relation -> has_date, hypothesis -> 0.1 million for the fiscal year ended December 31, 2019, confidence -> 0.9083246, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 537, entity1_end -> 418, entity2_begin -> 521, entity2 -> FISCAL_YEAR, chunk1 -> 0.1 million, sentence -> 2}, []}                                            |
|{category, 14455, 14696, has_date, {entity1_begin -> 849, relation -> has_date, hypothesis -> 1.5 million for the fiscal year ended December 31, 2020, confidence -> 0.75281376, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 914, entity1_end -> 859, entity2_begin -> 898, entity2 -> FISCAL_YEAR, chunk1 -> 1.5 million, sentence -> 5}, []}                                         |
|{category, 14697, 14938, has_date, {entity1_begin -> 849, relation -> has_date, hypothesis -> 1.5 million for the fiscal year ended December 31, 2019, confidence -> 0.8073463, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 970, entity1_end -> 859, entity2_begin -> 954, entity2 -> FISCAL_YEAR, chunk1 -> 1.5 million, sentence -> 5}, []}                                          |
|{category, 4874, 5113, has_date, {entity1_begin -> 42, relation -> has_date, hypothesis -> 0.5 million for the fiscal year ended December 31, 2020, confidence -> 0.71575713, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 106, entity1_end -> 52, entity2_begin -> 90, entity2 -> FISCAL_YEAR, chunk1 -> 0.5 million, sentence -> 0}, []}                                              |
|{category, 6908, 7115, profit_increase_to, {entity1_begin -> 172, relation -> profit_increase_to, hypothesis -> Services revenue to 24.5 million, confidence -> 0.85972106, nli_prediction -> entail, entity1 -> PROFIT_INCREASE, syntactic_distance -> undefined, chunk2 -> 24.5 million, entity2_end -> 295, entity1_end -> 187, entity2_begin -> 284, entity2 -> AMOUNT, chunk1 -> Services revenue, sentence -> 1}, []}                                          |
|{category, 5594, 5833, has_date, {entity1_begin -> 59, relation -> has_date, hypothesis -> 0.7 million for the fiscal year ended December 31, 2019, confidence -> 0.7484568, nli_prediction -> entail, entity1 -> AMOUNT, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 169, entity1_end -> 69, entity2_begin -> 153, entity2 -> FISCAL_YEAR, chunk1 -> 0.7 million, sentence -> 0}, []}                                              |
|{category, 7326, 7546, has_date, {entity1_begin -> 199, relation -> has_date, hypothesis -> 4 for the fiscal year ended December 31, 2020, confidence -> 0.8412763, nli_prediction -> entail, entity1 -> PERCENTAGE, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 275, entity1_end -> 199, entity2_begin -> 259, entity2 -> FISCAL_YEAR, chunk1 -> 4, sentence -> 1}, []}                                                            |
|{category, 9472, 9734, has_date, {entity1_begin -> 424, relation -> has_date, hypothesis -> 2 for the fiscal year ended December 31, 2020, confidence -> 0.8046481, nli_prediction -> entail, entity1 -> PERCENTAGE, syntactic_distance -> undefined, chunk2 -> December 31, 2020, entity2_end -> 481, entity1_end -> 424, entity2_begin -> 465, entity2 -> FISCAL_YEAR, chunk1 -> 2, sentence -> 2}, []}                                                            |
|{category, 9735, 9997, has_date, {entity1_begin -> 424, relation -> has_date, hypothesis -> 2 for the fiscal year ended December 31, 2019, confidence -> 0.8485106, nli_prediction -> entail, entity1 -> PERCENTAGE, syntactic_distance -> undefined, chunk2 -> December 31, 2019, entity2_end -> 537, entity1_end -> 424, entity2_begin -> 521, entity2 -> FISCAL_YEAR, chunk1 -> 2, sentence -> 2}, []}                                                            |
|{category, 691, 916, profit_decline_by_per, {entity1_begin -> 0, relation -> profit_decline_by_per, hypothesis -> License fees revenue decreased by a 40 to, confidence -> 0.9948003, nli_prediction -> entail, entity1 -> PROFIT_DECLINE, syntactic_distance -> undefined, chunk2 -> 40, entity2_end -> 32, entity1_end -> 19, entity2_begin -> 31, entity2 -> PERCENTAGE, chunk1 -> License fees revenue, sentence -> 0}, []}                                      |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
only showing top 20 rows

{%- endcapture -%}

{%- capture model_python_legal -%}

document_assembler = (
    nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
)
tokenizer = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")


tokenClassifier = (
    legal.BertForTokenClassification.pretrained(
        "legner_obligations", "en", "legal/models"
    )
    .setInputCols("document", "token")
    .setOutputCol("ner")
    .setCaseSensitive(True)
)

ner_converter = (
    nlp.NerConverter()
    .setInputCols(["document", "token", "ner"])
    .setOutputCol("ner_chunk")
)

re_model = (
    legal.ZeroShotRelationExtractionModel.pretrained(
        "legre_zero_shot", "en", "legal/models"
    )
    .setInputCols(["ner_chunk", "document"])
    .setOutputCol("relations")
)

re_model.setRelationalCategories(
    {
        "should_provide": [
            "{OBLIGATION_SUBJECT} will provide {OBLIGATION}",
            "{OBLIGATION_SUBJECT} should provide {OBLIGATION}",
        ],
        "commits_with": [
            "{OBLIGATION_SUBJECT} to {OBLIGATION_INDIRECT_OBJECT}",
            "{OBLIGATION_SUBJECT} with {OBLIGATION_INDIRECT_OBJECT}",
        ],
        "commits_to": ["{OBLIGATION_SUBJECT} commits to {OBLIGATION}"],
        "agree_to": ["{OBLIGATION_SUBJECT} agrees to {OBLIGATION}"],
    }
)

pipeline = nlp.Pipeline(
    stages=[document_assembler, tokenizer, tokenClassifier, ner_converter, re_model]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)
light_model = nlp.LightPipeline(model)

import pandas as pd

def get_relations_df(results, col="relations"):
    rel_pairs = []
    for i in range(len(results)):
        for rel in results[i][col]:
            rel_pairs.append(
                (
                    rel.result,
                    rel.metadata["entity1"],
                    rel.metadata["entity1_begin"],
                    rel.metadata["entity1_end"],
                    rel.metadata["chunk1"],
                    rel.metadata["entity2"],
                    rel.metadata["entity2_begin"],
                    rel.metadata["entity2_end"],
                    rel.metadata["chunk2"],
                    rel.metadata["confidence"],
                )
            )
    rel_df = pd.DataFrame(
        rel_pairs,
        columns=[
            "relation",
            "entity1",
            "entity1_begin",
            "entity1_end",
            "chunk1",
            "entity2",
            "entity2_begin",
            "entity2_end",
            "chunk2",
            "confidence",
        ],
    )
    return rel_df

sample_text = """This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties")."""

result = light_model.fullAnnotate(sample_text)
rel_df = get_relations_df(result)
rel_df[rel_df["relation"] != "no_rel"]
|             relation | entity1 | entity1_begin | entity1_end |                              chunk1 | entity2 | entity2_begin | entity2_end |                  chunk2 | confidence |
|---------------------:|--------:|--------------:|------------:|------------------------------------:|--------:|--------------:|------------:|------------------------:|-----------:|
|             dated_as |     DOC |             5 |          35 |     INTELLECTUAL PROPERTY AGREEMENT | EFFDATE |            69 |          85 |       December 31, 2018 | 0.98433626 |
|            signed_by |     DOC |             5 |          35 |     INTELLECTUAL PROPERTY AGREEMENT |   PARTY |           141 |         163 | Armstrong Flooring, Inc | 0.60404813 |
|            has_alias |   PARTY |           141 |         163 |             Armstrong Flooring, Inc |   ALIAS |           192 |         197 |                  Seller | 0.96357507 |
|            has_alias |   PARTY |           205 |         221 |                   AFI Licensing LLC |   ALIAS |           263 |         271 |               Licensing |  0.9546678 |
|            has_alias |   PARTY |           315 |         330 |                    AHF Holding, Inc |   ALIAS |           611 |         615 |                   Party |  0.5387175 |
|            has_alias |   PARTY |           315 |         330 |                    AHF Holding, Inc |   ALIAS |           641 |         647 |                 Parties |  0.5387175 |
| has_collective_alias |   ALIAS |           399 |         403 |                               Buyer |   ALIAS |           611 |         615 |                   Party |  0.5539445 |
| has_collective_alias |   ALIAS |           399 |         403 |                               Buyer |   ALIAS |           641 |         647 |                 Parties |  0.5539445 |
|            has_alias |   PARTY |           411 |         445 | Armstrong Hardwood Flooring Company |   ALIAS |           478 |         484 |                 Company | 0.92106056 |
|            has_alias |   PARTY |           411 |         445 | Armstrong Hardwood Flooring Company |   ALIAS |           611 |         615 |                   Party | 0.58123946 |
|            has_alias |   PARTY |           411 |         445 | Armstrong Hardwood Flooring Company |   ALIAS |           641 |         647 |                 Parties | 0.58123946 |
| has_collective_alias |   ALIAS |           505 |         509 |                               Buyer |   ALIAS |           516 |         529 |          Buyer Entities | 0.63492435 |
| has_collective_alias |   ALIAS |           505 |         509 |                               Buyer |   ALIAS |           611 |         615 |                   Party |  0.6483803 |
| has_collective_alias |   ALIAS |           505 |         509 |                               Buyer |   ALIAS |           641 |         647 |                 Parties |  0.6483803 |
| has_collective_alias |   ALIAS |           516 |         529 |                      Buyer Entities |   ALIAS |           611 |         615 |                   Party |  0.6970743 |
| has_collective_alias |   ALIAS |           516 |         529 |                      Buyer Entities |   ALIAS |           641 |         647 |                 Parties |  0.6970743 |

{%- endcapture -%}


{%- capture model_scala_medical -%}

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("tokens")

val sentencer = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentences")

val embeddings = WordEmbeddingsModel
  .pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("embeddings")

val posTagger = PerceptronModel
  .pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("posTags")

val nerTagger = MedicalNerModel
  .pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens", "embeddings"))
  .setOutputCol("nerTags")

val nerConverter = new NerConverter()
  .setInputCols(Array("sentences", "tokens", "nerTags"))
  .setOutputCol("nerChunks")

val dependencyParser = DependencyParserModel
  .pretrained("dependency_conllu", "en")
  .setInputCols(Array("document", "posTags", "tokens"))
  .setOutputCol("dependencies")

val reNerFilter = new RENerChunksFilter()
  .setRelationPairs(Array("problem-test","problem-treatment"))
  .setMaxSyntacticDistance(4)
  .setDocLevelRelations(false)
  .setInputCols(Array("nerChunks", "dependencies"))
  .setOutputCol("RENerChunks")

val re = ZeroShotRelationExtractionModel
  .load("/tmp/spark_sbert_zero_shot")
  .setRelationalCategories(
    Map(
      "CURE" -> Array("{TREATMENT} cures {PROBLEM}."),
      "IMPROVE" -> Array("{TREATMENT} improves {PROBLEM}.", "{TREATMENT} cures {PROBLEM}."),
      "REVEAL" -> Array("{TEST} reveals {PROBLEM}.")
      ))
  .setPredictionThreshold(0.9f)
  .setMultiLabel(false)
  .setInputCols(Array("sentences", "RENerChunks"))
  .setOutputCol("relations)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentencer,
    tokenizer,
    embeddings,
    posTagger,
    nerTagger,
    nerConverter,
    dependencyParser,
    reNerFilter,
    re))

val model = pipeline.fit(Seq("").toDS.toDF("text"))
val results = model.transform(
  Seq("Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer.").toDS.toDF("text"))

results
  .selectExpr("EXPLODE(relations) as relation")
  .selectExpr("relation.result", "relation.metadata.confidence")
  .show(truncate = false)

+-------+----------+
|result |confidence|
+-------+----------+
|REVEAL |0.9760039 |
|IMPROVE|0.98819494|
|IMPROVE|0.9929625 |
+-------+----------+


{%- endcapture -%}


{%- capture model_api_link -%}
[ZeroShotRelationExtractionModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/finance/graph/relation_extraction/ZeroShotRelationExtractionModel.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ZeroShotRelationExtractionModel](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/re/zero_shot_relation_extraction/index.html#sparknlp_jsl.annotator.re.zero_shot_relation_extraction.ZeroShotRelationExtractionModel)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_python_finance=model_python_finance
model_python_legal=model_python_legal
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
