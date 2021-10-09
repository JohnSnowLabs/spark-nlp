{%- capture title -%}
MedicalNer
{%- endcapture -%}

{%- capture description -%}
This Named Entity recognition annotator allows to train generic NER model based on Neural Networks.

The architecture of the neural network is a Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.

For instantiated/pretrained models, see NerDLModel.

The training data should be a labeled Spark Dataset, in the format of [CoNLL](/docs/en/training#conll-dataset)
2003 IOB with `Annotation` type columns. The data should have columns of type `DOCUMENT, TOKEN, WORD_EMBEDDINGS` and an
additional label column of annotator type `NAMED_ENTITY`.
Excluding the label, this can be done with for example
- a [SentenceDetector](/docs/en/annotators#sentencedetector),
- a [Tokenizer](/docs/en/annotators#tokenizer) and
- a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings) with clinical embeddings
  (any [clinical word embeddings](https://nlp.johnsnowlabs.com/models?task=Embeddings) can be chosen).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

# First extract the prerequisites for the NerDLApproach
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

# Then the training can start
nerTagger = MedicalNerApproach()\
.setInputCols(["sentence", "token", "embeddings"])\
.setLabelColumn("label")\
.setOutputCol("ner")\
.setMaxEpochs(2)\
.setBatchSize(64)\
.setRandomSeed(0)\
.setVerbose(1)\
.setValidationSplit(0.2)\
.setEvaluationLogExtended(True) \
.setEnableOutputLogs(True)\
.setIncludeConfidence(True)\
.setOutputLogsPath('ner_logs')\
.setGraphFolder('medical_ner_graphs')\
.setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
tokenizer,
clinical_embeddings,
nerTagger
])

# We use the text and labels from the CoNLL dataset
conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.MedicalNerApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline

// First extract the prerequisites for the NerDLApproach
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

// Then the training can start
val nerTagger =new MedicalNerApproach()
.setInputCols(Array("sentence", "token", "embeddings"))
.setLabelColumn("label")
.setOutputCol("ner")
.setMaxEpochs(5)
.setLr(0.003f)
.setBatchSize(8)
.setRandomSeed(0)
.setVerbose(1)
.setEvaluationLogExtended(false)
.setEnableOutputLogs(false)
.setIncludeConfidence(true)


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

// We use the text and labels from the CoNLL dataset
val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture api_link -%}
[MedicalNerApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/MedicalNerApproach.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[MedicalNerApproach](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl.annotator.MedicalNerApproach.html)
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
