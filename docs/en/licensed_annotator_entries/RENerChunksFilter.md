{%- capture title -%}
RENerChunksFilter
{%- endcapture -%}

{%- capture description -%}
Filters and outputs combinations of relations between extracted entities, for further processing.
This annotator is especially useful to create inputs for the RelationExtractionDLModel.
{%- endcapture -%}

{%- capture input_anno -%}
CHUNK, DEPENDENCY
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Define pipeline stages to extract entities
documenter = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentencer = SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

tokenizer = Tokenizer() \
  .setInputCols(["sentences"]) \
  .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("embeddings")

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["sentences", "tokens"]) \
  .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en") \
  .setInputCols(["sentences", "pos_tags", "tokens"]) \
  .setOutputCol("dependencies")

clinical_ner_tagger = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models") \
  .setInputCols(["sentences", "tokens", "embeddings"]) \
  .setOutputCol("ner_tags")

ner_chunker = NerConverter() \
  .setInputCols(["sentences", "tokens", "ner_tags"]) \
  .setOutputCol("ner_chunks")

# Define the relation pairs and the filter
relationPairs = [
  "direction-external_body_part_or_region",
  "external_body_part_or_region-direction",
  "direction-internal_organ_or_component",
  "internal_organ_or_component-direction"
]

re_ner_chunk_filter = RENerChunksFilter() \
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

{%- capture scala_example -%}
// Define pipeline stages to extract entities
val documenter = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentencer = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val tokenizer = new Tokenizer()
  .setInputCols("sentences")
  .setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("sentences", "tokens")
  .setOutputCol("embeddings")

val pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols("sentences", "tokens")
  .setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en")
  .setInputCols("sentences", "pos_tags", "tokens")
  .setOutputCol("dependencies")

val clinical_ner_tagger = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
  .setInputCols("sentences", "tokens", "embeddings")
  .setOutputCol("ner_tags")

val ner_chunker = new NerConverter()
  .setInputCols("sentences", "tokens", "ner_tags")
  .setOutputCol("ner_chunks")

// Define the relation pairs and the filter
val relationPairs = Array("direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction")

val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols("ner_chunks", "dependencies")
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

{%- capture api_link -%}
[RENerChunksFilter](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RENerChunksFilter)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}