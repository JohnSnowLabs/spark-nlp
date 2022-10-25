/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, ParamsAndFeaturesWritable}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types._

import scala.util.Random

/** Trains a ClassifierDL for generic Multi-class Text Classification.
  *
  * ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text
  * classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built
  * inside TensorFlow and supports up to 100 classes.
  *
  * For instantiated/pretrained models, see [[ClassifierDLModel]].
  *
  * '''Notes''':
  *   - This annotator accepts a label column of a single item in either type of String, Int,
  *     Float, or Double.
  *   - [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
  *     [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
  *     [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] can be used for
  *     the `inputCol`.
  *
  * For extended examples of usage, see the Spark NLP Workshop
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/scala/training/Train%20Multi-Class%20Text%20Classification%20on%20News%20Articles.scala [1]]]
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb [2]]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLTestSpec.scala ClassifierDLTestSpec]].
  *
  * ==Example==
  * In this example, the training data `"sentiment.csv"` has the form of
  * {{{
  * text,label
  * This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
  * This was a terrible movie! The acting was bad really bad!,1
  * ...
  * }}}
  * Then traning can be done like so:
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
  * import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
  * import org.apache.spark.ml.Pipeline
  *
  * val smallCorpus = spark.read.option("header","true").csv("src/test/resources/classifier/sentiment.csv")
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val useEmbeddings = UniversalSentenceEncoder.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  *
  * val docClassifier = new ClassifierDLApproach()
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCol("category")
  *   .setLabelColumn("label")
  *   .setBatchSize(64)
  *   .setMaxEpochs(20)
  *   .setLr(5e-3f)
  *   .setDropout(0.5f)
  *
  * val pipeline = new Pipeline()
  *   .setStages(
  *     Array(
  *       documentAssembler,
  *       useEmbeddings,
  *       docClassifier
  *     )
  *   )
  *
  * val pipelineModel = pipeline.fit(smallCorpus)
  * }}}
  *
  * @see
  *   [[MultiClassifierDLApproach]] for multi-class classification
  * @see
  *   [[SentimentDLApproach]] for sentiment analysis
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class ClassifierDLApproach(override val uid: String)
    extends AnnotatorApproach[ClassifierDLModel]
    with ParamsAndFeaturesWritable
    with ClassifierEncoder {

  def this() = this(Identifiable.randomUID("ClassifierDL"))

  /** Trains TensorFlow model for multi-class text classification */
  override val description = "Trains TensorFlow model for multi-class text classification"

  /** Input annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)

  /** Output annotator type : CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: String = CATEGORY

  /** Dropout coefficient (Default: `0.5f`)
    *
    * @group param
    */
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")

  /** Dropout coefficient (Default: `0.5f`)
    *
    * @group setParam
    */
  def setDropout(dropout: Float): ClassifierDLApproach.this.type = set(this.dropout, dropout)

  /** Dropout coefficient (Default: `0.5f`)
    *
    * @group getParam
    */
  def getDropout: Float = $(this.dropout)

  setDefault(maxEpochs -> 10, lr -> 5e-3f, dropout -> 0.5f, batchSize -> 64)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): ClassifierDLModel = {

    val labelColType = dataset.schema($(labelColumn)).dataType
    require(
      labelColType == StringType | labelColType == IntegerType | labelColType == DoubleType | labelColType == FloatType | labelColType == LongType,
      s"The label column $labelColumn type is $labelColType and it's not compatible. " +
        s"Compatible types are StringType, IntegerType, DoubleType, LongType, or FloatType. ")

    val (trainDataset, trainLabels) = buildDatasetWithLabels(dataset, getInputCols(0))
    val settings = ClassifierDatasetEncoderParams(tags = trainLabels)
    val encoder = new ClassifierDatasetEncoder(settings)
    val trainInputs = extractInputs(encoder, trainDataset)

    var testEncoder: Option[ClassifierDatasetEncoder] = None
    val testInputs =
      if (!isDefined(testDataset)) None
      else {
        val testDataFrame = ResourceHelper.readSparkDataFrame($(testDataset))
        val (test, testLabels) = buildDatasetWithLabels(testDataFrame, getInputCols(0))
        val settings = ClassifierDatasetEncoderParams(tags = testLabels)
        testEncoder = Some(new ClassifierDatasetEncoder(settings))
        Option(extractInputs(testEncoder.get, test))
      }

    val tfWrapper: TensorflowWrapper = loadSavedModel()

    val classifier =
      try {
        val model = new TensorflowClassifier(
          tensorflow = tfWrapper,
          encoder,
          testEncoder,
          Verbose($(verbose)))
        if (isDefined(randomSeed)) {
          Random.setSeed($(randomSeed))
        }

        model.train(
          trainInputs,
          testInputs,
          trainLabels.length,
          lr = $(lr),
          batchSize = $(batchSize),
          dropout = $(dropout),
          endEpoch = $(maxEpochs),
          configProtoBytes = getConfigProtoBytes,
          validationSplit = $(validationSplit),
          evaluationLogExtended = $(evaluationLogExtended),
          enableOutputLogs = $(enableOutputLogs),
          outputLogsPath = $(outputLogsPath),
          uuid = this.uid)
        model
      } catch {
        case e: Exception =>
          throw e
      }

    val newWrapper = new TensorflowWrapper(
      TensorflowWrapper.extractVariablesSavedModel(
        tfWrapper.getTFSession(configProtoBytes = getConfigProtoBytes)),
      tfWrapper.graph)

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(
      dataset,
      $(inputCols),
      AnnotatorType.SENTENCE_EMBEDDINGS)

    val model = new ClassifierDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper
        .readZippedSavedModel("/classifier-dl", tags = Array("serve"), initAllTables = true)

    wrapper.variables = Variables(Array.empty[Array[Byte]], Array.empty[Byte])
    wrapper
  }
}

/** This is the companion object of [[ClassifierDLApproach]]. Please refer to that class for the
  * documentation.
  */
object ClassifierDLApproach extends DefaultParamsReadable[ClassifierDLApproach]
