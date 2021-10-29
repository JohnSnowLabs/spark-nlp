{%- capture title -%}
AssertionLogRegApproach
{%- endcapture -%}

{%- capture description -%}
Train a Assertion Model algorithm using a regression log model. 

The training data should have annotations columns of type `DOCUMENT`, `CHUNK`, `WORD_EMBEDDINGS`, the `label`column (The assertion status that you want to predict), the `start` (the start index for the term that has the assertion status),
the `end` column (the end index for the term that has the assertion status).This model use a deep learning to predict the entity.

Excluding the label, this can be done with for example
- a [SentenceDetector](/docs/en/annotators#sentencedetector),
- a [Chunk](https://nlp.johnsnowlabs.com/docs/en/annotators#chunker) ,
- a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings)
  (any word embeddings can be chosen, e.g. [BertEmbeddings](/docs/en/transformers#bertembeddings) for BERT based embeddings).

{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
ASSERTION
{%- endcapture -%}

{%- capture python_example -%}

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

tokenizer = Tokenizer().setInputCols("sentence").setOutputCol("token")

POSTag = PerceptronModel.pretrained() \
.setInputCols("sentence", "token") \
.setOutputCol("pos")

chunker = Chunker() \
.setInputCols(["pos", "sentence"]) \
.setOutputCol("chunk") \
.setRegexParsers(["(<NN>)+"])

pubmed = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings") \
.setCaseSensitive(False)

assertion_status = AssertionLogRegApproach() \
.setInputCols("sentence", "chunk", "embeddings") \
.setOutputCol("assertion") \
.setStartCol("start") \
.setEndCol("end") \
.setLabelCol("label") \
.setReg(0.01) \
.setBefore(11) \
.setAfter(13) \
.setEpochs(3) 


pipeline = Pipeline().setStages([
document_assembler,
sentence_detector,
tokenizer,
POSTag,
chunker,
pubmed,
assertion_status
])


conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture scala_example -%}
// This CoNLL dataset already includes the sentence, token, pos and label column with their respective annotator types.
// If a custom dataset is used, these need to be defined.

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{Chunker, Tokenizer}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotator.PerceptronModel
import com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel
import com.johnsnowlabs.nlp.annotator.NerCrfApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val POSTag = PerceptronModel
.pretrained()
.setInputCols("sentence", "token")
.setOutputCol("pos")

val chunker = new Chunker()
.setInputCols(Array("pos", "sentence"))
.setOutputCol("chunk")
.setRegexParsers(Array("(<NN>)+"))

val pubmed = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
.setCaseSensitive(false)

val assertion = new AssertionLogRegApproach()
.setLabelCol("label")
.setInputCols("document", "chunk", "embeddings")
.setOutputCol("assertion")
.setReg(0.01)
.setBefore(11)
.setAfter(13)
.setStartCol("start")
.setEndCol("end")

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentenceDetector,
tokenizer,
POSTag,
chunker,
pubmed,
assertion
))

datasetPath = "/../src/test/resources/rsAnnotations-1-120-random.csv"
train_data = SparkContextForTest.spark.read.option("header", "true").csv(path="file:///" + os.getcwd() + datasetPath)

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture api_link -%}
[AssertionLogRegApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegApproach.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[AssertionLogRegApproach](https://nlp.johnsnowlabs.comlicensed/api/python/reference/autosummary/sparknlp_jsl.annotator.AssertionLogRegApproach.html)

{%- endcapture -%}



{% include templates/licensed_training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
%}
