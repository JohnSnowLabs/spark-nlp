{%- capture title -%}
EntityChunkEmbeddings
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Weighted average embeddings of multiple named entities chunk annotations.

Entity Chunk Embeddings uses BERT Sentence embeddings to compute a weighted average vector represention of related entity chunks.  The input the model consists of chunks of recognized named entities. One or more entities are selected as target entities and for each of them a list of related entities is specified (if empty, all other entities are assumed to be related).

The model looks for chunks of the target entities and then tries to pair each target entity (e.g. DRUG)  with other related entities (e.g. DOSAGE, STRENGTH, FORM, etc). The criterion for pairing a target entity with another related entity is that they appear in the same sentence and the maximal syntactic distance is below a predefined threshold.

The relationship between target and related entities is one-to-many, meaning that if there multiple instances of the same target entity (e.g.) within a sentence, the model will map a related entity (e.g. DOSAGE) to at most one of the instances of the target entity. For example, if there is a sentence "The patient was given 125 mg of paracetamol and metformin", the model will pair "125 mg" to "paracetamol", but not to "metformin".

The output of the model is an average embeddings of the chunks of each of the target entities and their related entities. It is possible to specify a particular weight for each entity type.

An entity can be defined both as target a entity and as a related entity for some other target entity. For example, we may want to compute the embeddings of SYMPTOMs and their related entities, as well as the embeddings of DRUGs and their related entities, one of each is also SYMPTOM. In such cases, it is possible to use the TARGET_ENTITY:RELATED_ENTITY notation to specify the weight of an related entity (e.g. "DRUG:SYMPTOM" to set the weight of SYMPTOM when it appears as an related entity to target entity DRUG). The relative weights of entities for particular entity chunk embeddings are available in the annotations metadata.

This model is a subclass of `BertSentenceEmbeddings` and shares all parameters
with it. It can load any pretrained `BertSentenceEmbeddings` model.

The default model is `"sbiobert_base_cased_mli"` from `clinical/models`.
Other available models can be found at [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

{%- endcapture -%}

{%- capture model_input_anno -%}
DEPENDENCY, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_python_medical -%}
import sparknlp
from sparknlp.base import *
from sparknlp_jsl.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("documents")
sentence_detector = SentenceDetector() \
    .setInputCols("documents") \
    .setOutputCol("sentences")
tokenizer = Tokenizer() \
    .setInputCols("sentences") \
    .setOutputCol("tokens")
embeddings = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")
ner_model = MedicalNerModel()\
    .pretrained("ner_posology_large", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens", "embeddings"])\
    .setOutputCol("ner")
ner_converter = NerConverterInternal()\
    .setInputCols("sentences", "tokens", "ner")\
    .setOutputCol("ner_chunks")
pos_tager = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens")\
    .setOutputCol("pos_tags")
dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")
drug_chunk_embeddings = EntityChunkEmbeddings()\
    .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("drug_chunk_embeddings")\
    .setMaxSyntacticDistance(3)\
    .setTargetEntities({"DRUG": []})
    .setEntityWeights({"DRUG": 0.8, "STRENGTH": 0.2, "DOSAGE": 0.2, "FORM": 0.5})
sampleData = "The parient was given metformin 125 mg, 250 mg of coumadin and then one pill paracetamol"
data = SparkContextForTest.spark.createDataFrame([[sampleData]]).toDF("text")
pipeline = Pipeline().setStages([
    documenter,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter,
    pos_tager,
    dependency_parser,
    drug_chunk_embeddings])
results = pipeline.fit(data).transform(data)
results = results \
    .selectExpr("explode(drug_chunk_embeddings) AS drug_chunk") \
    .selectExpr("drug_chunk.result", "slice(drug_chunk.embeddings, 1, 5) AS drug_embedding") \
    .cache()
results.show(truncate=False)
+-----------------------------+-----------------------------------------------------------------+
|                       result|                                                  drug_embedding"|
+-----------------------------+-----------------------------------------------------------------+
|metformin 125 mg             |[-0.267413, 0.07614058, -0.5620966, 0.83838946, 0.8911504]       |
|250 mg coumadin              |[0.22319649, -0.07094894, -0.6885556, 0.79176235, 0.82672405]    |
|one pill paracetamol         |[-0.10939768, -0.29242, -0.3574444, 0.3981813, 0.79609615]       |
+-----------------------------+-----------------------------------------------------------------+
{%- endcapture -%}

{%- capture model_scala_medical -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.ner.{MedicalNerModel, NerConverterInternal}
import com.johnsnowlabs.nlp.annotators.embeddings.EntityChunkEmbeddings
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
   .setInputCol("text")
   .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
   .setInputCols("document")
   .setOutputCol("sentence")

val tokenizer = new Tokenizer()
   .setInputCols("sentence")
   .setOutputCol("tokens")

val wordEmbeddings = WordEmbeddingsModel
   .pretrained("embeddings_clinical", "en", "clinical/models")
   .setInputCols(Array("sentences", "tokens"))
   .setOutputCol("word_embeddings")

val nerModel = MedicalNerModel
   .pretrained("ner_posology_large", "en", "clinical/models")
   .setInputCols(Array("sentence", "tokens", "word_embeddings"))
   .setOutputCol("ner")

val nerConverter = new NerConverterInternal()
   .setInputCols("sentence", "tokens", "ner")
   .setOutputCol("ner_chunk")

val posTager = PerceptronModel
   .pretrained("pos_clinical", "en", "clinical/models")
   .setInputCols("sentences", "tokens")
   .setOutputCol("pos_tags")

val dependencyParser = DependencyParserModel
   .pretrained("dependency_conllu", "en")
   .setInputCols(Array("sentences", "pos_tags", "tokens"))
   .setOutputCol("dependencies")

val drugChunkEmbeddings = EntityChunkEmbeddings
   .pretrained("sbiobert_base_cased_mli","en","clinical/models")
   .setInputCols(Array("ner_chunks", "dependencies"))
   .setOutputCol("drug_chunk_embeddings")
   .setMaxSyntacticDistance(3)
   .setTargetEntities(Map("DRUG" -> List()))
   .setEntityWeights(Map[String, Float]("DRUG" -> 0.8f, "STRENGTH" -> 0.2f, "DOSAGE" -> 0.2f, "FORM" -> 0.5f))

val pipeline = new Pipeline()
     .setStages(Array(
         documentAssembler,
         sentenceDetector,
         tokenizer,
         wordEmbeddings,
         nerModel,
         nerConverter,
         posTager,
         dependencyParser,
         drugChunkEmbeddings))

val sampleText = "The patient was given metformin 125 mg, 250 mg of coumadin and then one pill paracetamol."

val testDataset = Seq("").toDS.toDF("text")
val result = pipeline.fit(emptyDataset).transform(testDataset)

result
   .selectExpr("explode(drug_chunk_embeddings) AS drug_chunk")
   .selectExpr("drug_chunk.result", "slice(drug_chunk.embeddings, 1, 5) AS drugEmbedding")
   .show(truncate=false)

+-----------------------------+-----------------------------------------------------------------+
|                       result|                                                    drugEmbedding|
+-----------------------------+-----------------------------------------------------------------+
|metformin 125 mg             |[-0.267413, 0.07614058, -0.5620966, 0.83838946, 0.8911504]       |
|250 mg coumadin              |[0.22319649, -0.07094894, -0.6885556, 0.79176235, 0.82672405]    |
|one pill paracetamol          |[-0.10939768, -0.29242, -0.3574444, 0.3981813, 0.79609615]      |
+-----------------------------+----------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[EntityChunkEmbeddingsModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/embeddings/EntityChunkEmbeddings.html)
{%- endcapture -%}


{%- capture model_python_api_link -%}
[EntityChunkEmbeddingsModel](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/embeddings/entity_chunk_embeddings/index.html#sparknlp_jsl.annotator.embeddings.entity_chunk_embeddings.EntityChunkEmbeddings)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
