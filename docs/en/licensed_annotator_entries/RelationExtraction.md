{%- capture title -%}
RelationExtraction
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Extracts and classifies instances of relations between named entities. For this, relation pairs
need to be defined with `setRelationPairs`, to specify between which entities the extraction should be done.

For pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Relation+Extraction) for available models.
{%- endcapture -%}

{%- capture model_input_anno -%}
WORD_EMBEDDINGS, POS, CHUNK, DEPENDENCY
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Relation Extraction between body parts
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

# Define the relations that are to be extracted
relationPairs = [
  "direction-external_body_part_or_region",
  "external_body_part_or_region-direction",
  "direction-internal_organ_or_component",
  "internal_organ_or_component-direction"
]

re_model = medical.RelationExtractionModel.pretrained("re_bodypart_directions", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"]) \
    .setOutputCol("relations") \
    .setRelationPairs(relationPairs) \
    .setMaxSyntacticDistance(4) \
    .setPredictionThreshold(0.9)

pipeline = Pipeline().setStages([
    documenter,
    sentencer,
    tokenizer,
    words_embedder,
    pos_tagger,
    clinical_ner_tagger,
    ner_chunker,
    dependency_parser,
    re_model
])

data = spark.createDataFrame([["MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia"]]).toDF("text")
result = pipeline.fit(data).transform(data)

# Show results
#
result.selectExpr("explode(relations) as relations")
 .select(
   "relations.metadata.chunk1",
   "relations.metadata.entity1",
   "relations.metadata.chunk2",
   "relations.metadata.entity2",
   "relations.result"
 )
 .where("result != 0")
 .show(truncate=False)

# Show results
result.selectExpr("explode(relations) as relations") \
  .select(
     "relations.metadata.chunk1",
     "relations.metadata.entity1",
     "relations.metadata.chunk2",
     "relations.metadata.entity2",
     "relations.result"
  ).where("result != 0") \
  .show(truncate=False)
+------+---------+-------------+---------------------------+------+
|chunk1|entity1  |chunk2       |entity2                    |result|
+------+---------+-------------+---------------------------+------+
|upper |Direction|brain stem   |Internal_organ_or_component|1     |
|left  |Direction|cerebellum   |Internal_organ_or_component|1     |
|right |Direction|basil ganglia|Internal_organ_or_component|1     |
+------+---------+-------------+---------------------------+------+
{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Relation Extraction between body parts
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

// Define the relations that are to be extracted
val relationPairs = Array("direction-external_body_part_or_region",
                      "external_body_part_or_region-direction",
                      "direction-internal_organ_or_component",
                      "internal_organ_or_component-direction")

val re_model = medical.RelationExtractionModel.pretrained("re_bodypart_directions", "en", "clinical/models")
  .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
  .setOutputCol("relations")
  .setRelationPairs(relationPairs)
  .setMaxSyntacticDistance(4)
  .setPredictionThreshold(0.9f)

val pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_model
))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDF("text")
val result = pipeline.fit(data).transform(data)

// Show results
//
// result.selectExpr("explode(relations) as relations")
//  .select(
//    "relations.metadata.chunk1",
//    "relations.metadata.entity1",
//    "relations.metadata.chunk2",
//    "relations.metadata.entity2",
//    "relations.result"
//  )
//  .where("result != 0")
//  .show(truncate=false)
// +------+---------+-------------+---------------------------+------+
// |chunk1|entity1  |chunk2       |entity2                    |result|
// +------+---------+-------------+---------------------------+------+
// |upper |Direction|brain stem   |Internal_organ_or_component|1     |
// |left  |Direction|cerebellum   |Internal_organ_or_component|1     |
// |right |Direction|basil ganglia|Internal_organ_or_component|1     |
// +------+---------+-------------+---------------------------+------+
//
{%- endcapture -%}


{%- capture model_api_link -%}
[RelationExtractionModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionModel)
{%- endcapture -%}


{%- capture approach_description -%}
Trains a TensorFlow model for relation extraction. The Tensorflow graph in `.pb` format needs to be specified with
`setModelFile`. The result is a RelationExtractionModel.
To start training, see the parameters that need to be set in the Parameters section.
{%- endcapture -%}

{%- capture approach_input_anno -%}
WORD_EMBEDDINGS, POS, CHUNK, DEPENDENCY
{%- endcapture -%}

{%- capture approach_output_anno -%}
NONE
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import *
# Defining pipeline stages to extract entities first
documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("tokens")

embedder = nlp.WordEmbeddingsModel \
  .pretrained("embeddings_clinical", "en", "clinical/models") \
  .setInputCols(["document", "tokens"]) \
  .setOutputCol("embeddings")

posTagger = nlp.PerceptronModel \
  .pretrained("pos_clinical", "en", "clinical/models") \
  .setInputCols(["document", "tokens"]) \
  .setOutputCol("posTags")

nerTagger = nlp.MedicalNerModel \
  .pretrained("ner_events_clinical", "en", "clinical/models") \
  .setInputCols(["document", "tokens", "embeddings"]) \
  .setOutputCol("ner_tags")

nerConverter = nlp.NerConverter() \
  .setInputCols(["document", "tokens", "ner_tags"]) \
  .setOutputCol("nerChunks")

depencyParser = nlp.DependencyParserModel \
  .pretrained("dependency_conllu", "en") \
  .setInputCols(["document", "posTags", "tokens"]) \
  .setOutputCol("dependencies")

# Then define `RelationExtractionApproach` and training parameters
re = medical.RelationExtractionApproach() \
  .setInputCols(["embeddings", "posTags", "train_ner_chunks", "dependencies"]) \
  .setOutputCol("relations_t") \
  .setLabelColumn("target_rel") \
  .setEpochsNumber(300) \
  .setBatchSize(200) \
  .setLearningRate(0.001) \
  .setModelFile("path/to/graph_file.pb") \
  .setFixImbalance(True) \
  .setValidationSplit(0.05) \
  .setFromEntity("from_begin", "from_end", "from_label") \
  .setToEntity("to_begin", "to_end", "to_label")

finisher = nlp.Finisher() \
  .setInputCols(["relations_t"]) \
  .setOutputCols(["relations"]) \
  .setCleanAnnotations(False) \
  .setValueSplitSymbol(",") \
  .setAnnotationSplitSymbol(",") \
  .setOutputAsArray(False)

# Define complete pipeline and start training
pipeline = Pipeline(stages=[
    documentAssembler,
    tokenizer,
    embedder,
    posTagger,
    nerTagger,
    nerConverter,
    depencyParser,
    re,
    finisher])

model = pipeline.fit(trainData)

{%- endcapture -%}

{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
// Defining pipeline stages to extract entities first
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("document")
  .setOutputCol("tokens")

val embedder = nlp.WordEmbeddingsModel
  .pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens"))
  .setOutputCol("embeddings")

val posTagger = nlp.PerceptronModel
  .pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens"))
  .setOutputCol("posTags")

val nerTagger = medical.NerModel
  .pretrained("ner_events_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens", "embeddings"))
  .setOutputCol("ner_tags")

val nerConverter = new nlp.NerConverter()
  .setInputCols(Array("document", "tokens", "ner_tags"))
  .setOutputCol("nerChunks")

val depencyParser = nlp.DependencyParserModel
  .pretrained("dependency_conllu", "en")
  .setInputCols(Array("document", "posTags", "tokens"))
  .setOutputCol("dependencies")

// Then define `RelationExtractionApproach` and training parameters
val re = new medical.RelationExtractionApproach()
  .setInputCols(Array("embeddings", "posTags", "train_ner_chunks", "dependencies"))
  .setOutputCol("relations_t")
  .setLabelColumn("target_rel")
  .setEpochsNumber(300)
  .setBatchSize(200)
  .setlearningRate(0.001f)
  .setModelFile("path/to/graph_file.pb")
  .setFixImbalance(true)
  .setValidationSplit(0.05f)
  .setFromEntity("from_begin", "from_end", "from_label")
  .setToEntity("to_begin", "to_end", "to_label")

val finisher = new nlp.Finisher()
  .setInputCols(Array("relations_t"))
  .setOutputCols(Array("relations"))
  .setCleanAnnotations(false)
  .setValueSplitSymbol(",")
  .setAnnotationSplitSymbol(",")
  .setOutputAsArray(false)

// Define complete pipeline and start training
val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embedder,
    posTagger,
    nerTagger,
    nerConverter,
    depencyParser,
    re,
    finisher))

val model = pipeline.fit(trainData)

{%- endcapture -%}


{%- capture approach_api_link -%}
[RelationExtractionApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_scala_medical=approach_scala_medical
approach_api_link=approach_api_link
%}
