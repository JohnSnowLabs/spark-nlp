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

package com.johnsnowlabs.nlp.annotators.sentence_detector_dl

import com.johnsnowlabs.ml.ai.SentenceDetectorDL
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

case class Metrics(accuracy: Double, recall: Double, precision: Double, f1: Double)

/** Annotator that detects sentence boundaries using a deep learning approach.
  *
  * Instantiated Model of the
  * [[com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach SentenceDetectorDLApproach]].
  * Detects sentence boundaries using a deep learning approach.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val sentenceDL = SentenceDetectorDLModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentencesDL")
  * }}}
  * The default model is `"sentence_detector_dl"`, if no name is provided. For available
  * pretrained models please see the
  * [[https://sparknlp.org/models?task=Sentence+Detection Models Hub]].
  *
  * Each extracted sentence can be returned in an Array or exploded to separate rows, if
  * `explodeSentences` is set to `true`.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/multilingual/SentenceDetectorDL.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLSpec.scala SentenceDetectorDLSpec]].
  *
  * ==Example==
  * In this example, the normal `SentenceDetector` is compared to the `SentenceDetectorDLModel`.
  * In a pipeline, `SentenceDetectorDLModel` can be used as a replacement for the
  * `SentenceDetector`.
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentences")
  *
  * val sentenceDL = SentenceDetectorDLModel
  *   .pretrained("sentence_detector_dl", "en")
  *   .setInputCols("document")
  *   .setOutputCol("sentencesDL")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   sentenceDL
  * ))
  *
  * val data = Seq("""John loves Mary.Mary loves Peter
  *   Peter loves Helen .Helen loves John;
  *   Total: four people involved.""").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(sentences.result) as sentences").show(false)
  * +----------------------------------------------------------+
  * |sentences                                                 |
  * +----------------------------------------------------------+
  * |John loves Mary.Mary loves Peter\n     Peter loves Helen .|
  * |Helen loves John;                                         |
  * |Total: four people involved.                              |
  * +----------------------------------------------------------+
  *
  * result.selectExpr("explode(sentencesDL.result) as sentencesDL").show(false)
  * +----------------------------+
  * |sentencesDL                 |
  * +----------------------------+
  * |John loves Mary.            |
  * |Mary loves Peter            |
  * |Peter loves Helen .         |
  * |Helen loves John;           |
  * |Total: four people involved.|
  * +----------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach SentenceDetectorDLApproach]]
  *   for training a model yourself
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector SentenceDetector]] for non
  *   deep learning extraction
  * @param uid
  *   required uid for storing annotator to disk
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
class SentenceDetectorDLModel(override val uid: String)
    extends AnnotatorModel[SentenceDetectorDLModel]
    with HasSimpleAnnotate[SentenceDetectorDLModel]
    with HasStorageRef
    with ParamsAndFeaturesWritable
    with WriteTensorflowModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("SentenceDetectorDLModel"))

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: String = DOCUMENT

  var encoder = new SentenceDetectorDLEncoderParam(this, "Encoder", "Data encoder")

  def setEncoder(encoder: SentenceDetectorDLEncoder): SentenceDetectorDLModel.this.type =
    set(this.encoder, encoder)

  def getEncoder: SentenceDetectorDLEncoder = $(this.encoder)

  /** Model architecture (Default: `"cnn"`)
    *
    * @group param
    */
  var modelArchitecture =
    new Param[String](this, "modelArchitecture", "Model Architecture: one of (CNN)")

  /** Set architecture
    *
    * @group setParam
    */
  def setModel(modelArchitecture: String): SentenceDetectorDLModel.this.type =
    set(this.modelArchitecture, modelArchitecture)

  /** Get model architecture
    *
    * @group getParam
    */
  def getModel: String = $(this.modelArchitecture)

  /** Impossible penultimates (Default: `Array()`)
    *
    * @group param
    */
  val impossiblePenultimates =
    new StringArrayParam(this, "impossiblePenultimates", "Impossible penultimates")

  /** Length at which sentences will be forcibly split (Ignored if not set)
    *
    * @group param
    */

  val splitLength: IntParam =
    new IntParam(this, "splitLength", "length at which sentences will be forcibly split.")

  /** Set the minimum allowed length for each sentence (Default: `0`)
    *
    * @group param
    */

  val minLength =
    new IntParam(this, "minLength", "Set the minimum allowed length for each sentence")

  /** Set the maximum allowed length for each sentence (Ignored if not set)
    *
    * @group param
    */
  val maxLength =
    new IntParam(this, "maxLength", "Set the maximum allowed length for each sentence")

  /** A flag indicating whether to split sentences into different Dataset rows. Useful for higher
    * parallelism in fat rows (Default: `false`)
    *
    * @group getParam
    */
  val explodeSentences =
    new BooleanParam(this, "explodeSentences", "Split sentences in separate rows")

  /** Whether to only utilize custom bounds for sentence detection (Default: `false`)
    *
    * @group param
    */
  val useCustomBoundsOnly = new BooleanParam(
    this,
    "useCustomBoundsOnly",
    "whether to only utilize custom bounds for sentence detection")

  /** Characters used to explicitly mark sentence bounds (Default: None)
    *
    * @group param
    */
  val customBounds: StringArrayParam = new StringArrayParam(
    this,
    "customBounds",
    "characters used to explicitly mark sentence bounds")

  /** Length at which sentences will be forcibly split
    * @group setParam
    */
  def setSplitLength(value: Int): this.type = set(splitLength, value)

  /** Length at which sentences will be forcibly split
    * @group getParam
    */
  def getSplitLength: Int = $(splitLength)

  /** Set the minimum allowed length for each sentence
    * @group setParam
    */
  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }

  /** Get the minimum allowed length for each sentence
    * @group getParam
    */
  def getMinLength: Int = $(minLength)

  /** Set the maximum allowed length for each sentence
    * @group setParam
    */
  def setMaxLength(value: Int): this.type = {
    require(
      value >= $ {
        minLength
      },
      "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }

  /** Get the maximum allowed length for each sentence
    * @group getParam
    */
  def getMaxLength: Int = $(maxLength)

  /** Set impossible penultimates
    *
    * @group setParam
    */
  def setImpossiblePenultimates(
      impossiblePenultimates: Array[String]): SentenceDetectorDLModel.this.type =
    set(this.impossiblePenultimates, impossiblePenultimates)

  /** Get impossible penultimates
    *
    * @group getParam
    */
  def getImpossiblePenultimates: Array[String] = $(this.impossiblePenultimates)

  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat
    * rows. Defaults to false.
    *
    * @group setParam
    */
  def setExplodeSentences(value: Boolean): SentenceDetectorDLModel.this.type =
    set(this.explodeSentences, value)

  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat
    * rows. Defaults to false.
    *
    * @group getParam
    */
  def getExplodeSentences: Boolean = $(this.explodeSentences)

  /** Custom sentence separator text
    * @group setParam
    */
  def setCustomBounds(value: Array[String]): this.type = set(customBounds, value)

  /** Custom sentence separator text
    * @group getParam
    */
  def getCustomBounds: Array[String] = $(customBounds)

  /** Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false.
    * Needs customBounds.
    * @group setParam
    */
  def setUseCustomBoundsOnly(value: Boolean): this.type = set(useCustomBoundsOnly, value)

  /** Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false.
    * Needs customBounds.
    * @group getParam
    */
  def getUseCustomBoundsOnly: Boolean = $(useCustomBoundsOnly)

  setDefault(
    modelArchitecture -> "cnn",
    impossiblePenultimates -> Array(),
    explodeSentences -> false,
    minLength -> 0,
    maxLength -> Int.MaxValue,
    splitLength -> Int.MaxValue,
    useCustomBoundsOnly -> false,
    customBounds -> Array.empty[String])

  private var _tfClassifier: Option[Broadcast[SentenceDetectorDL]] = None

  def setupTFClassifier(spark: SparkSession, tfWrapper: TensorflowWrapper): this.type = {
    if (_tfClassifier.isEmpty) {
      _tfClassifier = Some(spark.sparkContext.broadcast(new SentenceDetectorDL(tfWrapper)))
    }
    this
  }

  def setupNew(spark: SparkSession, modelPath: String, vocabularyPath: String): this.type = {
    val encoder = new SentenceDetectorDLEncoder()
    encoder.loadVocabulary(vocabularyPath)
    setEncoder(encoder)

    val (wrapper, _) = TensorflowWrapper.read(modelPath)
    setupTFClassifier(spark, wrapper)
  }

  def getTFClassifier: SentenceDetectorDL = {
    require(_tfClassifier.isDefined, "TF model not setup.")
    _tfClassifier.get.value
  }

  def getMetrics(text: String, injectNewLines: Boolean = false): Metrics = {

    var nExamples = 0.0
    var nRecall = 0.0
    var nPrecision = 0.0

    var accuracy = 0.0
    var recall = 0.0
    var precision = 0.0

    var pText = text

    if (injectNewLines) {
      val nlShare = (text.split("\n").length / 10).toInt
      Array
        .fill(nlShare)(Random.nextInt(text.length - 10))
        .foreach(pos => {
          if (text(pos) != '\n' && text(pos + 1) != '\n' && text(pos - 1) != '\n') {
            pText = pText.slice(0, pos) + "\n" + pText.slice(pos + 1, pText.length - 1)
          }
        })
    } else {
      pText = text
    }

    getEncoder
      .getEOSPositions(pText)
      .foreach(ex => {
        val (pos, vector) = ex
        val output = getTFClassifier.predict(Array(vector))
        val posPrediction = output._1(0)
        val posActivation = output._2(0)

        val groundTruth = (
          (pos < (text.length - 1) && text(pos + 1) == '\n')
            || (text(pos) == '\n' && pos > 0 && (!Array('.', ':', '?', '!', ';').contains(
              text(pos - 1))))
        )

        val prediction = (posActivation > 0.5f)

        accuracy += (if (groundTruth == prediction) 1.0 else 0.0)
        nExamples += 1.0

        if (groundTruth) {
          recall += (if (groundTruth == prediction) 1.0 else 0.0)

          nRecall += 1.0
        }

        if (prediction) {
          precision += (if (groundTruth == prediction) 1.0 else 0.0)
          nPrecision += 1.0
        }
      })

    accuracy = (if (nExamples > 0) (accuracy / nExamples) else 1)
    recall = (if (nRecall > 0) (recall / nRecall) else 1)
    precision = (if (nPrecision > 0) (precision / nPrecision) else 1)

    Metrics(
      accuracy,
      recall,
      precision,
      2.0 * (if ((recall + precision) > 0.0) (recall * precision / (recall + precision))
             else 0.0))
  }

  def processText(
      text: String,
      processCustomBounds: Boolean = true): Iterator[(Int, Int, String)] = {

    if (processCustomBounds) {
      var sentences = Array("")
      var sentenceStarts = Array(0)
      var currentPos = 0
      text.zipWithIndex.foreach(x => {
        val boundary = $(customBounds).find(b => sentences(currentPos).matches(".*" + b + "$"))
        if (boundary.isDefined) {
//          sentences(currentPos) = sentences(currentPos).dropRight(boundary.get.length)
          sentences = sentences ++ Array("")
          sentenceStarts = sentenceStarts ++ Array(x._2)
          currentPos += 1
        }
        if (!(sentences(currentPos).isEmpty && getEncoder.getSkipChars.contains(x._1)))
          sentences(currentPos) = sentences(currentPos) + x._1
        else
          sentenceStarts(currentPos) = sentenceStarts(currentPos) + 1
      })
      return if ($(useCustomBoundsOnly)) {
        sentences.zip(sentenceStarts).map(x => (x._2, x._2 + x._1.length, x._1)).toIterator
      } else {
        sentences
          .zip(sentenceStarts)
          .flatMap(x => {
            processText(x._1, false).map(s => (s._1 + x._2, s._2 + x._2, s._3))
          })
          .toIterator
      }

    }

    var startPos = 0
    val skipChars = getEncoder.getSkipChars

    val sentences = getEncoder
      .getEOSPositions(text, getImpossiblePenultimates)
      .map(ex => {
        val (pos, vector) = ex
        val output = getTFClassifier.predict(Array(vector))
        val posActivation = output._2(0)
        (pos, posActivation)
      })
      .filter(ex => ex._2 > 0.5f)
      .map(_._1)
      .map(eos => {

        while ((startPos < eos) && skipChars.contains(text(startPos))) {
          startPos += 1
        }

        val endPos = if (skipChars.contains(text(eos))) eos else eos + 1
        val s = (startPos, eos, text.slice(startPos, endPos))

        startPos = eos + 1

        s

      })

    sentences ++ (if (startPos < text.length)
                    Array((startPos, text.length, text.slice(startPos, text.length))).toIterator
                  else
                    Array().toIterator)
  }

  private def truncateSentence(sentence: String, maxLength: Int): Array[String] = {
    var currentLength = 0
    val allSentences = ArrayBuffer.empty[String]
    val currentSentence = ArrayBuffer.empty[String]

    def addWordToSentence(word: String): Unit = {

      /** Adds +1 because of the space joining words */
      currentLength += word.length + 1
      currentSentence.append(word)
    }

    sentence
      .split(" ")
      .foreach(word => {
        if (currentLength + word.length > maxLength) {
          allSentences.append(currentSentence.mkString(" "))
          currentSentence.clear()
          currentLength = 0
          addWordToSentence(word)
        } else {
          addWordToSentence(word)
        }
      })

    /** add leftovers */
    allSentences.append(currentSentence.mkString(" "))
    allSentences.toArray
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val documents = annotations.filter(_.annotatorType == DOCUMENT)
    val outputAnnotations = ArrayBuffer[Annotation]()

    documents.foreach(doc => {
      var sentenceNo = 0
      processText(doc.result).foreach(posSentence => {

        if (posSentence._3.trim.nonEmpty) {
          var sentenceBegin = posSentence._1

          truncateSentence(posSentence._3, getSplitLength).foreach(splitSentence => {

            outputAnnotations.append(
              new Annotation(
                annotatorType = AnnotatorType.DOCUMENT,
                begin = sentenceBegin,
                end = sentenceBegin + splitSentence.length - 1,
                result = splitSentence,
                metadata = doc.metadata ++ mutable.Map("sentence" -> sentenceNo.toString)))
            sentenceBegin += splitSentence.length
            sentenceNo += 1
          })
        }
      })
      if ((sentenceNo == 0) && (doc.end > doc.begin)) {
        outputAnnotations.append(
          new Annotation(
            annotatorType = AnnotatorType.DOCUMENT,
            begin = doc.begin,
            end = doc.end,
            result = doc.result,
            metadata = doc.metadata ++ mutable.Map("sentence" -> sentenceNo.toString)))
      }
    })

    outputAnnotations
      .filter(anno => anno.result.length >= getMinLength && anno.result.length <= getMaxLength)
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {

    import org.apache.spark.sql.functions.{array, col, explode}

    if ($(explodeSentences)) {
      dataset
        .select(dataset.columns.filterNot(_ == getOutputCol).map(col) :+ explode(
          col(getOutputCol)).as("_tmp"): _*)
        .withColumn(
          getOutputCol,
          array(col("_tmp"))
            .as(getOutputCol, dataset.schema.fields.find(_.name == getOutputCol).get.metadata))
        .drop("_tmp")
    } else dataset
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    writeTensorflowModel(
      path,
      spark,
      getTFClassifier.getTFModel,
      "_genericclassifier",
      SentenceDetectorDLModel.tfFile)
  }
}

trait ReadsSentenceDetectorDLGraph
    extends ParamsAndFeaturesReadable[SentenceDetectorDLModel]
    with ReadTensorflowModel {

  override val tfFile = "generic_classifier_tensorflow"

  def readSentenceDetectorDLGraph(
      instance: SentenceDetectorDLModel,
      path: String,
      spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_genericclassifier")
    instance.setupTFClassifier(spark, tf)
  }

  addReader(readSentenceDetectorDLGraph)
}

trait ReadablePretrainedSentenceDetectorDL
    extends ParamsAndFeaturesReadable[SentenceDetectorDLModel]
    with HasPretrained[SentenceDetectorDLModel] {

  override val defaultModelName: Some[String] = Some("sentence_detector_dl")

  /** Java compliant-overrides */
  override def pretrained(): SentenceDetectorDLModel = super.pretrained()

  override def pretrained(name: String): SentenceDetectorDLModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): SentenceDetectorDLModel =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): SentenceDetectorDLModel = super.pretrained(name, lang, remoteLoc)
}

object SentenceDetectorDLModel
    extends ReadsSentenceDetectorDLGraph
    with ReadablePretrainedSentenceDetectorDL
