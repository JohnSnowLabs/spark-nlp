/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.ml.tensorflow.wrap.TFWrapper
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowSentenceDetectorDL, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasPretrained, HasSimpleAnnotate, ParamsAndFeaturesReadable, ParamsAndFeaturesWritable}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, Map}
import scala.util.Random

case class Metrics(accuracy: Double, recall: Double, precision: Double, f1: Double)

/** Annotator that detects sentence boundaries using a deep learning approach.
 *
 * Instantiated Model of the [[com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach SentenceDetectorDLApproach]].
 * Detects sentence boundaries using a deep learning approach.
 *
 * Pretrained models can be loaded with `pretrained` of the companion object:
 * {{{
 * val sentenceDL = SentenceDetectorDLModel.pretrained()
 *   .setInputCols("document")
 *   .setOutputCol("sentencesDL")
 * }}}
 * The default model is `"sentence_detector_dl"`, if no name is provided.
 * For available pretrained models please see the [[https://nlp.johnsnowlabs.com/models?task=Sentence+Detection Models Hub]].
 *
 * Each extracted sentence can be returned in an Array or exploded to separate rows,
 * if `explodeSentences` is set to `true`.
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb Spark NLP Workshop]]
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLSpec.scala SentenceDetectorDLSpec]].
 *
 * ==Example==
 * In this example, the normal `SentenceDetector` is compared to the `SentenceDetectorDLModel`. In a pipeline,
 * `SentenceDetectorDLModel` can be used as a replacement for the `SentenceDetector`.
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
 * @see [[com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach SentenceDetectorDLApproach]] for training a model yourself
 * @see [[com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector SentenceDetector]] for non deep learning extraction
 * @param uid required uid for storing annotator to disk
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
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
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 */
class SentenceDetectorDLModel(override val uid: String)
  extends AnnotatorModel[SentenceDetectorDLModel] with HasSimpleAnnotate[SentenceDetectorDLModel]
    with HasStorageRef
    with ParamsAndFeaturesWritable
    with WriteTensorflowModel {

  def this() = this(Identifiable.randomUID("SentenceDetectorDLModel"))

  /** Output annotator type : DOCUMENT
   *
   * @group anno
   * */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
   *
   * @group anno
   * */
  override val outputAnnotatorType: String = DOCUMENT

  var encoder = new SentenceDetectorDLEncoderParam(this, "Encoder", "Data encoder")

  def setEncoder(encoder: SentenceDetectorDLEncoder): SentenceDetectorDLModel.this.type = set(this.encoder, encoder)

  def getEncoder: SentenceDetectorDLEncoder = $(this.encoder)


  /** Model architecture (Default: `"cnn"`)
   *
   * @group param
   * */
  var modelArchitecture = new Param[String](this, "modelArchitecture", "Model Architecture: one of (CNN)")


  /** Set architecture
   *
   * @group setParam
   * */
  def setModel(modelArchitecture: String): SentenceDetectorDLModel.this.type = set(this.modelArchitecture, modelArchitecture)

  /** Get model architecture
   *
   * @group getParam
   * */
  def getModel: String = $(this.modelArchitecture)

  /** Impossible penultimates (Default: `Array()`)
   *
   * @group param
   * */
  val impossiblePenultimates = new StringArrayParam(this, "impossiblePenultimates", "Impossible penultimates")

  /** whether to only utilize custom bounds for sentence detection */
  val useCustomBoundsOnly = new BooleanParam(this, "useCustomBoundsOnly", "whether to only utilize custom bounds for sentence detection")


  /** characters used to explicitly mark sentence bounds */
  val customBounds: StringArrayParam = new StringArrayParam(this, "customBounds", "characters used to explicitly mark sentence bounds")

  /** Set impossible penultimates
   *
   * @group setParam
   * */
  def setImpossiblePenultimates(impossiblePenultimates: Array[String]):
  SentenceDetectorDLModel.this.type = set(this.impossiblePenultimates, impossiblePenultimates)

  /** Get impossible penultimates
   *
   * @group getParam
   * */
  def getImpossiblePenultimates: Array[String] = $(this.impossiblePenultimates)

  /** A flag indicating whether to split sentences into different Dataset rows. Useful for higher parallelism in
   * fat rows (Default: `false`)
   *
   * @group getParam
   * */
  def explodeSentences = new BooleanParam(this, "explodeSentences", "Split sentences in separate rows")


  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.
   *
   * @group setParam
   * */
  def setExplodeSentences(value: Boolean): SentenceDetectorDLModel.this.type = set(this.explodeSentences, value)


  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.
   *
   * @group getParam
   * */
  def getExplodeSentences: Boolean = $(this.explodeSentences)

  setDefault(
    modelArchitecture -> "cnn",
    impossiblePenultimates -> Array(),
    explodeSentences -> false
  )

  private var _tfClassifier: Option[Broadcast[TensorflowSentenceDetectorDL]] = None

  def setupTFClassifier(spark: SparkSession, tfWrapper: TFWrapper[_]): this.type = {
    if (_tfClassifier.isEmpty) {
      _tfClassifier = Some(
        spark.sparkContext.broadcast(
          new TensorflowSentenceDetectorDL(
            tfWrapper
          )
        )
      )
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


  def getTFClassifier: TensorflowSentenceDetectorDL = {
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
      Array.fill(nlShare)(Random.nextInt(text.length - 10)).foreach(pos => {
        if (text(pos) != '\n' && text(pos + 1) != '\n' && text(pos - 1) != '\n') {
          pText = pText.slice(0, pos) + "\n" + pText.slice(pos + 1, pText.length - 1)
        }
      })
    } else {
      pText = text
    }


    getEncoder.getEOSPositions(pText).foreach(ex => {
      val (pos, vector) = ex
      val output = getTFClassifier.predict(Array(vector))
      val posPrediction = output._1(0)
      val posActivation = output._2(0)

      val groundTruth = (
        (pos < (text.length - 1) && text(pos + 1) == '\n')
          || (text(pos) == '\n' && pos > 0 && (!Array('.', ':', '?', '!', ';').contains(text(pos - 1))))
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
      2.0 * (if ((recall + precision) > 0.0) (recall * precision / (recall + precision)) else 0.0)
    )
  }

  def processText(text: String): Iterator[(Int, Int, String)] = {

    var startPos = 0
    val skipChars = getEncoder.getSkipChars

    val sentences = getEncoder.getEOSPositions(text, getImpossiblePenultimates).map(ex => {
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

    sentences ++ (
      if (startPos < text.length)
        Array((startPos, text.length, text.slice(startPos, text.length))).toIterator
      else
        Array().toIterator)
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val documents = annotations.filter(_.annotatorType == DOCUMENT)
    val outputAnnotations = ArrayBuffer[Annotation]()

    documents.foreach(doc => {
      var sentenceNo = 0
      processText(doc.result).foreach(posSentence => {

        if (posSentence._3.trim.nonEmpty) {
          outputAnnotations.append(
            new Annotation(
              annotatorType = AnnotatorType.DOCUMENT,
              begin = posSentence._1,
              end = posSentence._2,
              result = posSentence._3,
              metadata = mutable.Map(
                "sentence" -> sentenceNo.toString
              ))
          )
          sentenceNo += 1
        }
      })
      if ((sentenceNo == 0) && (doc.end > doc.begin)) {
        outputAnnotations.append(
          new Annotation(
            annotatorType = AnnotatorType.DOCUMENT,
            begin = doc.begin,
            end = doc.end,
            result = doc.result,
            metadata = mutable.Map(
              "sentence" -> sentenceNo.toString
            ))
        )
      }
    })

    outputAnnotations
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {

    import org.apache.spark.sql.functions.{array, col, explode}

    if ($(explodeSentences)) {
      dataset
        .select(
          dataset.columns.filterNot(_ == getOutputCol).map(col) :+ explode(col(getOutputCol)).as("_tmp"): _*)
        .withColumn(
          getOutputCol,
          array(col("_tmp")).as(
            getOutputCol,
            dataset.schema.fields.find(_.name == getOutputCol).get.metadata))
        .drop("_tmp")
    }

    else dataset
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    writeTensorflowModel(
      path, spark, getTFClassifier.getTFModel, "_genericclassifier", SentenceDetectorDLModel.tfFile)
  }
}

trait ReadsSentenceDetectorDLGraph extends ParamsAndFeaturesReadable[SentenceDetectorDLModel] with ReadTensorflowModel {

  override val tfFile = "generic_classifier_tensorflow"

  def readSentenceDetectorDLGraph(instance: SentenceDetectorDLModel, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_genericclassifier")
    instance.setupTFClassifier(spark, tf)
  }

  addReader(readSentenceDetectorDLGraph)
}

trait ReadablePretrainedSentenceDetectorDL
  extends ParamsAndFeaturesReadable[SentenceDetectorDLModel] with HasPretrained[SentenceDetectorDLModel] {

  override val defaultModelName: Some[String] = Some("sentence_detector_dl")

  /** Java compliant-overrides */
  override def pretrained(): SentenceDetectorDLModel = super.pretrained()

  override def pretrained(name: String): SentenceDetectorDLModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): SentenceDetectorDLModel = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): SentenceDetectorDLModel = super.pretrained(name, lang, remoteLoc)
}

object SentenceDetectorDLModel
  extends ReadsSentenceDetectorDLGraph
    with ReadablePretrainedSentenceDetectorDL
