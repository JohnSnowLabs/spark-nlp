{%- capture title -%}
RelationExtractionApproach
{%- endcapture -%}

{%- capture description -%}
Trains a Relation Extraction Model to predict attributes and relations for entities in a sentence.

Relation Extraction is the key component for building relation knowledge graphs, and it is of crucial significance to natural language 
processing applications such as structured search, sentiment analysis, question answering, and summarization.

The dataset will be a csv with the following that contains the following columns (`sentence`,`chunk1`,`firstCharEnt1`,`lastCharEnt1`,`label1`,`chunk2`,`firstCharEnt2`,`lastCharEnt2`,`label2`,`rel`),

This annotator can be don with for example:
Excluding the rel, this can be done with for example
- a [SentenceDetector](/docs/en/annotators#sentencedetector),
- a [Tokenizer](/docs/en/annotators#tokenizer) and
- a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings)
  (any word embeddings can be chosen, e.g. [BertEmbeddings](/docs/en/transformers#bertembeddings) for BERT based embeddings).
- a Chunk can be created using the `firstCharEnt1`, `lastCharEnt1`,`chunk1`, `label1` columns and `firstCharEnt2`, `lastCharEnt2`, `chunk2`,  `label2` `columns`
- 


An example of that dataset can be found in the following link [i2b2_clinical_dataset](https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/i2b2_clinical_rel_dataset.csv)

```
sentence,chunk1,firstCharEnt1,lastCharEnt1,label1,chunk2,firstCharEnt2,lastCharEnt2,label2,rel						
Previous studies have reported the association of prodynorphin (PDYN) promoter polymorphism with temporal lobe epilepsy (TLE) susceptibility, but the results remain inconclusive.,PDYN,64,67,GENE,epilepsy,111,118,PHENOTYPE,0						
The remaining cases, clinically similar to XLA, are autosomal recessive agammaglobulinemia (ARA).,XLA,43,45,GENE,autosomal recessive,52,70,PHENOTYPE,0						
YAP/TAZ have been reported to be highly expressed in malignant tumors.,YAP,19,21,GENE,tumors,82,87,PHENOTYPE,0						

```
Apart from that, no additional training data is needed.
{%- endcapture -%}

{%- capture input_anno -%}
WORD_EMBEDDINGS, POS, CHUNK, DEPENDENCY
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture python_example -%}


import functools
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from sparknlp.base import


annotationType = T.StructType([
T.StructField('annotatorType', T.StringType(), False),
T.StructField('begin', T.IntegerType(), False),
T.StructField('end', T.IntegerType(), False),
T.StructField('result', T.StringType(), False),
T.StructField('metadata', T.MapType(T.StringType(), T.StringType()), False),
T.StructField('embeddings', T.ArrayType(T.FloatType()), False)
])


@F.udf(T.ArrayType(annotationType))
def createTrainAnnotations(begin1, end1, begin2, end2, chunk1, chunk2, label1, label2):
    entity1 = sparknlp.annotation.Annotation("chunk", begin1, end1, chunk1, {'entity': label1.upper(), 'sentence': '0'}, [])
    entity2 = sparknlp.annotation.Annotation("chunk", begin2, end2, chunk2, {'entity': label2.upper(), 'sentence': '0'}, [])    
        
    entity1.annotatorType = "chunk"
    entity2.annotatorType = "chunk"
    return [entity1, entity2]

data = spark.read.option("header","true").format("csv").load("i2b2_clinical_rel_dataset.csv")


data = data
    .withColumn("begin1i", F.expr("cast(firstCharEnt1 AS Int)"))
    .withColumn("end1i", F.expr("cast(lastCharEnt1 AS Int)"))
    .withColumn("begin2i", F.expr("cast(firstCharEnt2 AS Int)"))
    .withColumn("end2i", F.expr("cast(lastCharEnt2 AS Int)"))
    .where("begin1i IS NOT NULL")
    .where("end1i IS NOT NULL")
    .where("begin2i IS NOT NULL")
    .where("end2i IS NOT NULL")
    .withColumn(
    "train_ner_chunks",
    createTrainAnnotations(
        "begin1i", "end1i", "begin2i", "end2i", "chunk1", "chunk2", "label1", "label2"
    ).alias("train_ner_chunks", metadata={'annotatorType': "chunk"}))



documentAssembler = DocumentAssembler() \
    .setInputCol("sentence") \
    .setOutputCol("sentences")


tokenizer = Tokenizer() \
    .setInputCols("sentences") \
    .setOutputCol("token")

words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(["sentences", "tokens"])
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(["sentences", "tokens"])
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(["sentences", "pos_tags", "tokens"])
    .setOutputCol("dependencies")

reApproach = RelationExtractionApproach()
    .setInputCols(["embeddings", "pos_tags", "train_ner_chunks", "dependencies"])
    .setOutputCol("relations")
    .setLabelColumn("rel")
    .setEpochsNumber(70)
    .setBatchSize(200)
    .setDropout(0.5)
    .setLearningRate(0.001)
    .setModelFile("/content/RE_in1200D_out20.pb")
    .setFixImbalance(True)
    .setFromEntity("begin1i", "end1i", "label1")
    .setToEntity("begin2i", "end2i", "label2")
    .setOutputLogsPath('/content')

train_pipeline = Pipeline(stages=[
            documenter,
            tokenizer,
            words_embedder,
            pos_tagger,
            dependency_parser,
            reApproach
])
rel_model = train_pipeline.fit(data)

{%- endcapture -%}

{%- capture scala_example -%}

import com.johnsnowlabs.nlp.{DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.{MedicalNerModel, NerConverter}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
package com.johnsnowlabs.nlp.annotators.re.RelationExtractionApproach()
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._


val data = spark.read.option("header",true).csv("src/test/resources/re/gene_hpi.csv").limit(10)



def createTrainAnnotations = udf {
 ( begin1:Int, end1:Int, begin2:Int, end2:Int, chunk1:String, chunk2:String, label1:String, label2:String) => {

    val an1 =   Annotation(CHUNK,begin1,end1,chunk1,Map("entity" -> label1.toUpperCase,"sentence" -> "0"))
    val an2 =   Annotation(CHUNK,begin2,end2,chunk2,Map("entity" -> label2.toUpperCase,"sentence" -> "0"))
    Seq(an1,an2)
 }

}
val metadataBuilder: MetadataBuilder = new MetadataBuilder()
val meta = metadataBuilder.putString("annotatorType", CHUNK).build()

val dataEncoded =  data
.withColumn("begin1i", expr("cast(firstCharEnt1 AS Int)"))
.withColumn("end1i", expr("cast(lastCharEnt1 AS Int)"))
.withColumn("begin2i", expr("cast(firstCharEnt2 AS Int)"))
.withColumn("end2i", expr("cast(lastCharEnt2 AS Int)"))
.where("begin1i IS NOT NULL")
.where("end1i IS NOT NULL")
.where("begin2i IS NOT NULL")
.where("end2i IS NOT NULL")
.withColumn(
  "train_ner_chunks",
  createTrainAnnotations(
    col("begin1i"), col("end1i"), col("begin2i"), col("end2i"), col("chunk1"), col("chunk2"), col("label1"), col("label2")
  ).as("train_ner_chunks",meta))

val documentAssembler = new DocumentAssembler()
  .setInputCol("sentence")
  .setOutputCol("sentences")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentences"))
  .setOutputCol("tokens")

val embedder = ParallelDownload(WordEmbeddingsModel
  .pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "tokens"))
  .setOutputCol("embeddings"))

val posTagger = ParallelDownload(PerceptronModel
  .pretrained("pos_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens"))
  .setOutputCol("posTags"))

val nerTagger = ParallelDownload(MedicalNerModel
  .pretrained("ner_events_clinical", "en", "clinical/models")
  .setInputCols(Array("sentences", "tokens", "embeddings"))
  .setOutputCol("ner_tags"))

val nerConverter = new NerConverter()
  .setInputCols(Array("sentences", "tokens", "ner_tags"))
  .setOutputCol("nerChunks")

val depencyParser = ParallelDownload(DependencyParserModel
  .pretrained("dependency_conllu", "en")
  .setInputCols(Array("sentences", "posTags", "tokens"))
  .setOutputCol("dependencies"))

val re = new RelationExtractionApproach()
  .setInputCols(Array("embeddings", "posTags", "train_ner_chunks", "dependencies"))
  .setOutputCol("rel")
  .setLabelColumn("target_rel")
  .setEpochsNumber(30)
  .setBatchSize(200)
  .setlearningRate(0.001f)
  .setValidationSplit(0.05f)
  .setFromEntity("begin1i", "end1i", "label1")
  .setToEntity("end2i", "end2i", "label2")



val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embedder,
    posTagger,
    nerTagger,
    nerConverter,
    depencyParser,
    re).parallelDownload)

    val model = pipeline.fit(dataEncoded)

{%- endcapture -%}

{%- capture api_link -%}
[RelationExtractionApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionApproach.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[RelationExtractionApproach](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl.annotator.RelationExtractionApproach.html)
{%- endcapture -%}



{% include templates/training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
%}
