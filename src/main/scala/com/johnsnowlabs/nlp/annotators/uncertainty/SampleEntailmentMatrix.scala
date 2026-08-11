/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators.uncertainty

import com.johnsnowlabs.ml.ai.NliClassification
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/** Computes a bidirectional-entailment matrix over a row's sampled LLM answers, using a BERT
  * sequence-classification model trained on NLI.
  *
  * This is the faithful-to-the-literature alternative to `LLMUncertaintyEstimator`'s default
  * `similarityBackend="embeddings"`: [[https://arxiv.org/abs/2302.09664 Kuhn et al. 2023]]'s
  * Semantic Entropy clusters samples by checking whether each pair of samples entails the other
  * (in both directions), rather than by embedding similarity.
  *
  * This is a plumbing annotator, like [[MarsTokenImportance]]: it does not itself produce an
  * uncertainty score, it only attaches an `entailment_matrix` metadata field (a row-major N x N
  * JSON array of entailment probabilities, entry `[i][j]` = P(sample i entails sample j)) that
  * `LLMUncertaintyEstimator` reads when `setSimilarityBackend("nli")` is set.
  *
  * ==Cost warning==
  * Scoring all ordered pairs of N samples needs N*(N-1) model calls (the diagonal, self-
  * entailment, is trivially `1.0` and skipped) - this grows fast. `maxSamplesForNli` (default
  * `10`, so up to 90 calls per row) guards against silently issuing very large batches; raise it
  * explicitly if you really want more samples clustered this way.
  *
  * ==No pretrained model is published yet==
  * `pretrained()` has no working hub model: Spark NLP's `.pretrained()` deserializes a model in
  * the format the *calling class itself* wrote it in (here, an ONNX file under
  * `sample_entailment_matrix_onnx`), and no BERT NLI checkpoint has ever been published to the
  * hub in that exact format - `bert_base_cased_zero_shot_classifier_xnli` and similar exist as
  * `BertForZeroShotClassification` models (onnx file `bert_classification_onnx`), which download
  * fine but then fail to deserialize as a `SampleEntailmentMatrix`. Use `loadSavedModel` with a
  * self-exported model instead. A verified-working recipe: export
  * [[https://huggingface.co/textattack/bert-base-uncased-MNLI textattack/bert-base-uncased-MNLI]]
  * to ONNX (`torch.onnx.export`, `dynamo=False`), and lay it out as
  * {{{
  * <model_dir>/model.onnx
  * <model_dir>/assets/vocab.txt    (tokenizer.get_vocab(), one wordpiece per line, by id)
  * <model_dir>/assets/labels.txt   (one label per line, by id)
  * }}}
  * `labels.txt`'s order is model-specific and not always the textbook GLUE MNLI convention -
  * this checkpoint's `config.json` has no `id2label` at all (`LABEL_0/1/2` placeholders), and its
  * actual trained order, confirmed empirically against unambiguous probe sentences, is
  * `contradiction, entailment, neutral` (`0, 1, 2`). Then:
  * {{{
  * val entailment = SampleEntailmentMatrix.loadSavedModel("<model_dir>", spark)
  *   .setInputCols("completions")
  *   .setOutputCol("entailment")
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio setParam  3
  * @groupprio getParam  4
  */
class SampleEntailmentMatrix(override val uid: String)
    extends AnnotatorModel[SampleEntailmentMatrix]
    with HasBatchedAnnotate[SampleEntailmentMatrix]
    with WriteOnnxModel
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("SAMPLE_ENTAILMENT_MATRIX"))

  /** Input Annotator Type: DOCUMENT (the sampled completions to cluster)
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Vocabulary used to encode words to wordpiece ids
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Labels used to decode predicted class ids; must include one starting with "entail" (case
    * insensitive)
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels").setProtected()

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /** Max combined (premise + hypothesis) sequence length to process (Default: `512`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max combined sequence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "BERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** Maximum number of sampled completions to cluster per row (Default: `10`); guards against
    * silently issuing a very large number of pairwise NLI calls (`n * (n - 1)` for `n` samples)
    *
    * @group param
    */
  val maxSamplesForNli = new IntParam(
    this,
    "maxSamplesForNli",
    "Maximum number of samples to cluster per row (guards against n*(n-1) blowup)")

  /** @group setParam */
  def setMaxSamplesForNli(value: Int): this.type = {
    require(value >= 1, "maxSamplesForNli must be at least 1")
    set(maxSamplesForNli, value)
  }

  /** @group getParam */
  def getMaxSamplesForNli: Int = $(maxSamplesForNli)

  private var _model: Option[Broadcast[NliClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, onnxWrapper: OnnxWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new NliClassification(onnxWrapper, $$(vocabulary), $$(labels), getCaseSensitive)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: NliClassification = _model.get.value

  override def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 512,
    maxSamplesForNli -> 10,
    caseSensitive -> true)

  /** Computes the row-major N x N entailment-probability matrix for one row's samples, using
    * `1.0` on the diagonal (self-entailment) without a model call.
    */
  private def entailmentMatrixJson(texts: Array[String]): String = {
    val n = texts.length
    val matrix: Array[Array[Float]] = Array.tabulate(n, n) { (i, j) =>
      if (i == j) 1.0f
      else getModelIfNotSet.entailmentProbability(texts(i), texts(j), getMaxSentenceLength)
    }
    compact(render(matrix.toSeq.map(row => row.toSeq.map(_.toDouble))))
  }

  /** Attaches the same row-level `entailment_matrix` to every sampled completion in each row.
    *
    * @param batchedAnnotations
    *   one `Array[Annotation]` per row: that row's sampled completions
    * @return
    *   one output annotation per input annotation (preserving its text and position), each
    *   carrying the full N x N `entailment_matrix` for that row
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map { rowAnnotations =>
      val n = rowAnnotations.length
      if (n == 0) Seq.empty[Annotation]
      else {
        require(
          n <= getMaxSamplesForNli,
          s"$uid: row has $n sampled completions, but maxSamplesForNli is $getMaxSamplesForNli " +
            s"(clustering $n samples needs ${n * (n - 1)} pairwise NLI calls). Lower " +
            "AutoGGUFModel.setNumSamples(n), or raise setMaxSamplesForNli explicitly if you " +
            "really want to cluster this many samples this way.")
        val texts = rowAnnotations.map(_.result)
        val matrixJson = entailmentMatrixJson(texts)
        rowAnnotations.map { annotation =>
          new Annotation(
            outputAnnotatorType,
            annotation.begin,
            annotation.end,
            annotation.result,
            annotation.metadata + ("entailment_matrix" -> matrixJson))
        }.toSeq
      }
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeOnnxModel(
      path,
      spark,
      getModelIfNotSet.onnxWrapper,
      "_sample_entailment_matrix",
      SampleEntailmentMatrix.onnxFile)
  }
}

trait ReadablePretrainedSampleEntailmentMatrix
    extends ParamsAndFeaturesReadable[SampleEntailmentMatrix]
    with HasPretrained[SampleEntailmentMatrix] {
  override val defaultModelName: Some[String] = Some("bert_base_cased_zero_shot_classifier_xnli")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): SampleEntailmentMatrix = super.pretrained()

  override def pretrained(name: String): SampleEntailmentMatrix = super.pretrained(name)

  override def pretrained(name: String, lang: String): SampleEntailmentMatrix =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): SampleEntailmentMatrix =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadSampleEntailmentMatrixModel extends ReadOnnxModel {
  this: ParamsAndFeaturesReadable[SampleEntailmentMatrix] =>

  override val onnxFile: String = "sample_entailment_matrix_onnx"

  def readModel(instance: SampleEntailmentMatrix, path: String, spark: SparkSession): Unit = {
    val onnxWrapper = readOnnxModel(path, spark, "sample_entailment_matrix_onnx")
    instance.setModelIfNotSet(spark, onnxWrapper)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): SampleEntailmentMatrix = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    require(
      detectedEngine == ONNX.name,
      s"SampleEntailmentMatrix only supports ONNX models. $notSupportedEngineError")

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
    val labelsMap = loadTextAsset(localModelPath, "labels.txt").zipWithIndex.toMap
    require(
      labelsMap.keys.exists(_.toLowerCase.startsWith("entail")),
      "This annotator requires a model trained on NLI data with a label starting with " +
        s"'entail' (case insensitive) in its labels.txt. Current labels: ${labelsMap.keys
            .mkString(", ")}")

    val onnxWrapper =
      OnnxModelStaging.readWithDistinctName(spark, localModelPath, "sample_entailment_matrix")

    new SampleEntailmentMatrix()
      .setVocabulary(vocabs)
      .setLabels(labelsMap)
      .setModelIfNotSet(spark, onnxWrapper)
  }
}

/** This is the companion object of [[SampleEntailmentMatrix]]. Please refer to that class for the
  * documentation.
  */
object SampleEntailmentMatrix
    extends ReadablePretrainedSampleEntailmentMatrix
    with ReadSampleEntailmentMatrixModel
