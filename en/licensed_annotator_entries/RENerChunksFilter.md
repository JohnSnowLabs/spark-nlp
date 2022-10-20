{%- capture title -%}
RENerChunksFilter
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Filters and outputs combinations of relations between extracted entities, for further processing.
This annotator is especially useful to create inputs for the RelationExtractionDLModel.
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, DEPENDENCY
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Define pipeline stages to extract entities
documenter = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentencer = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentences"]) \
  .setOutputCol("tokens")

words_embedder = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("embeddings")

pos_tagger = nlp.PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("pos_tags")

dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en") \
  .setInputCols(["sentences", "pos_tags", "tokens"]) \
  .setOutputCol("dependencies")

clinical_ner_tagger = medical.NerModel.pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models") \
  .setInputCols(["sentences", "tokens", "embeddings"]) \
  .setOutputCol("ner_tags")

ner_chunker = nlp.NerConverter() \
  .setInputCols(["sentences", "tokens", "ner_tags"]) \
  .setOutputCol("ner_chunks")

# Define the relation pairs and the filter
relationPairs = [
  "direction-external_body_part_or_region",
  "external_body_part_or_region-direction",
  "direction-internal_organ_or_component",
  "internal_organ_or_component-direction"
]

re_ner_chunk_filter = medical.RENerChunksFilter() \
  .setInputCols(["ner_chunks", "dependencies"]) \
  .setOutputCol("re_ner_chunks") \
  .setMaxSyntacticDistance(4) \
  .setRelationPairs(["internal_organ_or_component-direction"])

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter
])

data = spark.createDataFrame([["MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia"]]).toDF("text")
result = trained_pipeline.fit(data).transform(data)

# Show results
result.selectExpr("explode(re_ner_chunks) as re_chunks") \
  .selectExpr("re_chunks.begin", "re_chunks.result", "re_chunks.metadata.entity", "re_chunks.metadata.paired_to") \
  .show(6, truncate=False)
+-----+-------------+---------------------------+---------+
|begin|result       |entity                     |paired_to|
+-----+-------------+---------------------------+---------+
|35   |upper        |Direction                  |41       |
|41   |brain stem   |Internal_organ_or_component|35       |
|35   |upper        |Direction                  |59       |
|59   |cerebellum   |Internal_organ_or_component|35       |
|35   |upper        |Direction                  |81       |
|81   |basil ganglia|Internal_organ_or_component|35       |
+-----+-------------+---------------------------+---------+
{%- endcapture -%}


{%- capture model_python_legal -%}
from johnsnowlabs import * 
# Define pipeline stages to extract entities
documenter = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentencer = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentences"]) \
  .setOutputCol("tokens")

words_embedder = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("embeddings")

pos_tagger = nlp.PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("pos_tags")

dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en") \
  .setInputCols(["sentences", "pos_tags", "tokens"]) \
  .setOutputCol("dependencies")

ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias", "en", "legal/models")\
  .setInputCols(["sentence", "token", "embedding"])\
  .setOutputCol("ner")

ner_chunker = nlp.NerConverter() \
  .setInputCols(["sentences", "tokens", "ner"]) \
  .setOutputCol("ner_chunks")

# Define the relation pairs and the filter
relationPairs = [
  "direction-external_body_part_or_region",
  "external_body_part_or_region-direction",
  "direction-internal_organ_or_component",
  "internal_organ_or_component-direction"
]

re_ner_chunk_filter = legal.RENerChunksFilter() \
  .setInputCols(["ner_chunks", "dependencies"]) \
  .setOutputCol("re_ner_chunks") \
  .setMaxSyntacticDistance(4) \
  .setRelationPairs(["internal_organ_or_component-direction"])

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  dependency_parser,
  ner_model,
  ner_chunker,
  re_ner_chunk_filter
])
{%- endcapture -%}


{%- capture model_python_finance -%}
from johnsnowlabs import * 
# Define pipeline stages to extract entities
documenter = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentencer = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentences"]) \
  .setOutputCol("tokens")

words_embedder = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("embeddings")

pos_tagger = nlp.PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("pos_tags")

dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en") \
  .setInputCols(["sentences", "pos_tags", "tokens"]) \
  .setOutputCol("dependencies")

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")

ner_chunker = nlp.NerConverter() \
  .setInputCols(["sentences", "tokens", "ner"]) \
  .setOutputCol("ner_chunks")

# Define the relation pairs and the filter
relationPairs = [
  "direction-external_body_part_or_region",
  "external_body_part_or_region-direction",
  "direction-internal_organ_or_component",
  "internal_organ_or_component-direction"
]

re_ner_chunk_filter = finance.RENerChunksFilter() \
  .setInputCols(["ner_chunks", "dependencies"]) \
  .setOutputCol("re_ner_chunks") \
  .setMaxSyntacticDistance(4) \
  .setRelationPairs(["internal_organ_or_component-direction"])

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  dependency_parser,
  ner_model,
  ner_chunker,
  re_ner_chunk_filter
])
{%- endcapture -%}


{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Define pipeline stages to extract entities
val documenter = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentencer = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentences")
  .setOutputCol("tokens")

val words_embedder = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("embeddings")

val pos_tagger = nlp.PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("pos_tags")

val dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en")
  .setInputCols(Array("sentences", "pos_tags", "tokens"))
  .setOutputCol("dependencies")

val clinical_ner_tagger = medical.NerModel.pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
  .setInputCols(Array("sentences", "tokens", "embeddings"))
  .setOutputCol("ner_tags")

val ner_chunker = new nlp.NerConverter()
  .setInputCols(Array("sentences", "tokens", "ner_tags"))
  .setOutputCol("ner_chunks")

// Define the relation pairs and the filter
val relationPairs = Array("direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction")

val re_ner_chunk_filter = new medical.RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setOutputCol("re_ner_chunks")
    .setMaxSyntacticDistance(4)
    .setRelationPairs(Array("internal_organ_or_component-direction"))

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter
))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDF("text")
val result = trained_pipeline.fit(data).transform(data)

// Show results
//
// result.selectExpr("explode(re_ner_chunks) as re_chunks")
//   .selectExpr("re_chunks.begin", "re_chunks.result", "re_chunks.metadata.entity", "re_chunks.metadata.paired_to")
//   .show(6, truncate=false)
// +-----+-------------+---------------------------+---------+
// |begin|result       |entity                     |paired_to|
// +-----+-------------+---------------------------+---------+
// |35   |upper        |Direction                  |41       |
// |41   |brain stem   |Internal_organ_or_component|35       |
// |35   |upper        |Direction                  |59       |
// |59   |cerebellum   |Internal_organ_or_component|35       |
// |35   |upper        |Direction                  |81       |
// |81   |basil ganglia|Internal_organ_or_component|35       |
// +-----+-------------+---------------------------+---------+
//
{%- endcapture -%}


{%- capture model_scala_legal -%}
from johnsnowlabs import * 
// Define pipeline stages to extract entities
val documenter = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentencer = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentences")
  .setOutputCol("tokens")

val words_embedder = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("embeddings")

val pos_tagger = nlp.PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("pos_tags")

val dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en")
  .setInputCols(Array("sentences", "pos_tags", "tokens"))
  .setOutputCol("dependencies")

val ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias", "en", "legal/models")
  .setInputCols(Array("sentence", "token", "embedding"))
  .setOutputCol("ner")

val ner_chunker = new nlp.NerConverter()
  .setInputCols(Array("sentences", "tokens", "ner"))
  .setOutputCol("ner_chunks")

// Define the relation pairs and the filter
val relationPairs = Array("direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction")

val re_ner_chunk_filter = new legal.RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setOutputCol("re_ner_chunks")
    .setMaxSyntacticDistance(4)
    .setRelationPairs(Array("internal_organ_or_component-direction"))

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  dependency_parser,
  ner_model,
  ner_chunker,
  re_ner_chunk_filter
))
{%- endcapture -%}


{%- capture model_scala_finance -%}
from johnsnowlabs import * 
// Define pipeline stages to extract entities
val documenter = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentencer = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentences")
  .setOutputCol("tokens")

val words_embedder = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("embeddings")

val pos_tagger = nlp.PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("pos_tags")

val dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en")
  .setInputCols(Array("sentences", "pos_tags", "tokens"))
  .setOutputCol("dependencies")

val ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val ner_chunker = new nlp.NerConverter()
  .setInputCols(Array("sentences", "tokens", "ner"))
  .setOutputCol("ner_chunks")

// Define the relation pairs and the filter
val relationPairs = Array("direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction")

val re_ner_chunk_filter = new finance.RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setOutputCol("re_ner_chunks")
    .setMaxSyntacticDistance(4)
    .setRelationPairs(Array("internal_organ_or_component-direction"))

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  dependency_parser,
  ner_model,
  ner_chunker,
  re_ner_chunk_filter
))
{%- endcapture -%}




{%- capture model_api_link -%}
[RENerChunksFilter](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RENerChunksFilter)
{%- endcapture -%}

{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_python_legal=model_python_legal
model_python_finance=model_python_finance
model_scala_medical=model_scala_medical
model_scala_legal=model_scala_legal
model_scala_finance=model_scala_finance
model_api_link=model_api_link%}
