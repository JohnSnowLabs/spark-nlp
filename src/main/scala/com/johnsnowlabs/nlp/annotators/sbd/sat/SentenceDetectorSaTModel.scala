/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators.sbd.sat

import com.johnsnowlabs.ml.ai.{SaT, SentenceSpan}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Spark NLP sentence detection annotator based on the wtpsplit / SaT transformer models.
  *
  * Supports `segment-any-text/sat-12l-sm` and `segment-any-text/sat-12l` (and any other SaT model
  * exported as ONNX with an XLM-R SentencePiece tokenizer).
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.sbd.sat.SentenceDetectorSaTModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val satModel = SentenceDetectorSaTModel
  *   .loadSavedModel("/path/to/sat-12l-sm", spark)
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *   .setThreshold(0.25f)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, satModel))
  * val result   = pipeline.fit(data).transform(data)
  * result.selectExpr("explode(sentence.result)").show(false)
  * }}}
  *
  * @param uid
  *   Required UID for Spark ML serialization.
  * @groupname anno Annotator types
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param 1
  * @groupprio anno 2
  * @groupprio setParam 4
  * @groupprio getParam 5
  */
class SentenceDetectorSaTModel(override val uid: String)
    extends AnnotatorModel[SentenceDetectorSaTModel]
    with HasBatchedAnnotate[SentenceDetectorSaTModel]
    with WriteOnnxModel
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("SENTENCE_DETECTOR_SAT"))

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val outputAnnotatorType: String = DOCUMENT

  /** Boundary probability threshold (Default: `0.25` for sat-12l-sm).
    *
    * A token boundary is emitted when sigmoid(logit) >= threshold. Typical values:
    *   - `sat-12l-sm` -> 0.25
    *   - `sat-12l` -> 0.025
    *   - LoRA merged -> 0.5
    *
    * @group param
    */
  val threshold: FloatParam =
    new FloatParam(
      this,
      "threshold",
      "Boundary probability threshold (default 0.25 for sat-12l-sm)")

  /** @group setParam */
  def setThreshold(value: Float): this.type = set(threshold, value)

  /** @group getParam */
  def getThreshold: Float = $(threshold)

  /** Number of real sub-word tokens per ONNX window (Default: `510`).
    *
    * XLM-R has a 512-position limit. Every window adds `<s>` + tokens + `</s>`, so the maximum
    * number of real tokens per window is 510.
    *
    * @group param
    */
  val blockSize: IntParam =
    new IntParam(this, "blockSize", "Real sub-word tokens per window (max 510 for XLM-R)")

  /** @group setParam */
  def setBlockSize(value: Int): this.type = {
    require(value >= 1 && value <= 510, "blockSize must be in [1, 510]")
    set(blockSize, value)
  }

  /** @group getParam */
  def getBlockSize: Int = $(blockSize)

  /** Number of tokens to advance between consecutive windows (Default: `256`).
    *
    * A smaller stride means more overlap and smoother boundary probabilities.
    *
    * @group param
    */
  val stride: IntParam =
    new IntParam(this, "stride", "Token stride between overlapping windows (default 256)")

  /** @group setParam */
  def setStride(value: Int): this.type = {
    require(value >= 1, "stride must be >= 1")
    set(stride, value)
  }

  /** @group getParam */
  def getStride: Int = $(stride)

  /** Number of windows to batch together in one ONNX call (Default: `8`).
    *
    * @group param
    */
  val satBatchSize: IntParam =
    new IntParam(this, "satBatchSize", "Number of windows per ONNX batch (default 8)")

  /** @group setParam */
  def setSatBatchSize(value: Int): this.type = {
    require(value >= 1, "satBatchSize must be >= 1")
    set(satBatchSize, value)
  }

  /** @group getParam */
  def getSatBatchSize: Int = $(satBatchSize)

  /** Window-overlap weighting strategy (Default: `"hat"`).
    *
    * Supported values:
    *   - `"hat"` - centre tokens get higher weight than edge tokens.
    *   - `"uniform"` - every token in a window has equal weight 1.0.
    *
    * @group param
    */
  val weighting: Param[String] =
    new Param[String](
      this,
      "weighting",
      "Overlap weighting: 'hat' (preferred) or 'uniform'",
      (v: String) => Seq("hat", "uniform").contains(v))

  /** @group setParam */
  def setWeighting(value: String): this.type = set(weighting, value)

  /** @group getParam */
  def getWeighting: String = $(weighting)

  /** Strip leading/trailing whitespace from each detected sentence (Default: `true`).
    *
    * @group param
    */
  val trimWhitespace: BooleanParam =
    new BooleanParam(this, "trimWhitespace", "Trim whitespace from sentence boundaries")

  /** @group setParam */
  def setTrimWhitespace(value: Boolean): this.type = set(trimWhitespace, value)

  /** @group getParam */
  def getTrimWhitespace: Boolean = $(trimWhitespace)

  /** Whether to split each detected sentence into its own Dataset row (Default: `true`).
    * @group param
    */
  val explodeSentences: BooleanParam =
    new BooleanParam(this, "explodeSentences", "Split sentences in separate rows")

  /** @group setParam */
  def setExplodeSentences(value: Boolean): this.type = set(explodeSentences, value)

  /** @group getParam */
  def getExplodeSentences: Boolean = $(explodeSentences)

  /** Minimum sentence length in characters (Default: `0` = no minimum).
    *
    * When `minSentenceLength` or [[maxSentenceLength]] is `> 0`, the model switches to
    * length-constrained (Viterbi) segmentation and the [[threshold]] is ignored: it instead finds
    * the globally highest-probability set of boundaries such that every sentence falls within
    * `[minSentenceLength, maxSentenceLength]` characters.
    *
    * @group param
    */
  val minSentenceLength: IntParam =
    new IntParam(this, "minSentenceLength", "Minimum sentence length in characters (0 = unset)")

  /** @group setParam */
  def setMinSentenceLength(value: Int): this.type = set(minSentenceLength, value)

  /** @group getParam */
  def getMinSentenceLength: Int = $(minSentenceLength)

  /** Maximum sentence length in characters (Default: `0` = no maximum).
    *
    * See [[minSentenceLength]]: setting either bound activates length-constrained segmentation
    * and disables the [[threshold]].
    *
    * @group param
    */
  val maxSentenceLength: IntParam =
    new IntParam(this, "maxSentenceLength", "Maximum sentence length in characters (0 = unset)")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = set(maxSentenceLength, value)

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  setDefault(
    threshold -> 0.25f,
    blockSize -> 510,
    stride -> 256,
    satBatchSize -> 8,
    weighting -> "hat",
    trimWhitespace -> true,
    explodeSentences -> false,
    minSentenceLength -> 0,
    maxSentenceLength -> 0,
    batchSize -> 4,
    engine -> ONNX.name)

  private var _model: Option[Broadcast[SaT]] = None

  /** Set the fully-initialised [[SaT]] inference object. Called by [[loadSavedModel]] and by the
    * Spark ML deserialization reader.
    */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrapper: OnnxWrapper,
      spp: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(new SaT(onnxWrapper, spp)))
    }
    this
  }

  def getModelIfNotSet: SaT =
    _model.getOrElse(throw new IllegalStateException("SaT model is not loaded.")).value

  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val satModel = getModelIfNotSet
    batchedAnnotations.map { annotations =>
      val docAnnotations = annotations.filter(_.annotatorType == DOCUMENT)
      if (docAnnotations.isEmpty) {
        Seq.empty[Annotation]
      } else {
        val results = scala.collection.mutable.ArrayBuffer.empty[Annotation]
        for (doc <- docAnnotations) {
          val text = doc.result
          if (text != null && text.nonEmpty) {
            val spans: Seq[SentenceSpan] = satModel.split(
              text = text,
              threshold = $(threshold),
              blockSize = $(blockSize),
              stride = $(stride),
              batchSize = $(satBatchSize),
              weighting = $(weighting),
              trimWhitespace = $(trimWhitespace),
              minSentenceLength = $(minSentenceLength),
              maxSentenceLength = $(maxSentenceLength))
            val docOffset = doc.begin
            for ((span, idx) <- spans.zipWithIndex) {
              results += Annotation(
                annotatorType = DOCUMENT,
                begin = docOffset + span.begin,
                end = docOffset + span.end,
                result = span.text,
                metadata = doc.metadata ++ Map(
                  "sentence" -> idx.toString,
                  "threshold" -> $(threshold).toString),
                embeddings = Array.emptyFloatArray)
            }
          }
        }
        results.toSeq
      }
    }
  }

  /** When `explodeSentences` is `true`, explode the output column so each sentence annotation
    * lands on its own Dataset row. Mirrors `SentenceDetectorDLModel.afterAnnotate`.
    */
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
    val m =
      _model.getOrElse(throw new IllegalStateException("Cannot save: model not loaded.")).value
    writeOnnxModel(path, spark, m.onnxWrapper, "_sat", SentenceDetectorSaTModel.onnxFile)
    writeSentencePieceModel(path, spark, m.spp, "_sat", SentenceDetectorSaTModel.sppFile)
  }
}

trait ReadablePretrainedSaTModel
    extends ParamsAndFeaturesReadable[SentenceDetectorSaTModel]
    with HasPretrained[SentenceDetectorSaTModel] {
  override val defaultModelName: Some[String] = Some("sat_12l_sm")
  override val defaultLang: String = "xx"
  override def pretrained(): SentenceDetectorSaTModel = super.pretrained()
  override def pretrained(name: String): SentenceDetectorSaTModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): SentenceDetectorSaTModel =
    super.pretrained(name, lang)
  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): SentenceDetectorSaTModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadSaTDLModel extends ReadOnnxModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[SentenceDetectorSaTModel] =>

  override val onnxFile: String = "sat_onnx"
  override val sppFile: String = "sat_spp"

  def readModel(instance: SentenceDetectorSaTModel, path: String, spark: SparkSession): Unit = {
    val onnxWrapper =
      readOnnxModel(path, spark, "_sat_onnx", zipped = true, useBundle = false, None)
    val spp = readSentencePieceModel(path, spark, "_sat_spp", sppFile)
    instance.setModelIfNotSet(spark, onnxWrapper, spp)
  }

  addReader(readModel)

  /** Load a SaT model from a local folder exported by the Hugging Face / ONNX export script.
    *
    * @param modelPath
    *   Local or remote (HDFS / S3 / GCS) path to the model folder.
    * @param spark
    *   Active SparkSession.
    * @return
    *   A ready-to-use [[SentenceDetectorSaTModel]].
    */
  def loadSavedModel(modelPath: String, spark: SparkSession): SentenceDetectorSaTModel = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    if (detectedEngine != ONNX.name)
      throw new IllegalArgumentException(
        s"SentenceDetectorSaTModel only supports ONNX, but detected engine: $detectedEngine. " +
          notSupportedEngineError)
    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")
    val onnxWrapper = OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
    val annotatorModel = new SentenceDetectorSaTModel()
    annotatorModel
      .set(annotatorModel.engine, ONNX.name)
      .setModelIfNotSet(spark, onnxWrapper, spModel)
  }
}

object SentenceDetectorSaTModel extends ReadablePretrainedSaTModel with ReadSaTDLModel
