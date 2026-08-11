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

import com.johnsnowlabs.ml.ai.{MarsClassification, MarsPhrase}
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

/** Computes MARS per-token importance weights for sampled LLM answers, given the question they
  * answer, using a BERT token-classification model ([[https://huggingface.co/duygunuryldz/MARS
  * duygunuryldz/MARS]] by default - [[https://arxiv.org/abs/2402.11756 Bakman et al. 2024]]).
  *
  * This is a plumbing annotator for [[LLMUncertaintyEstimator]]'s `mars` method: it does not
  * itself produce an uncertainty score, it only attaches a `token_importance` metadata field (a
  * JSON array of `{"begin", "end", "importance"}` character-offset spans into the answer) that
  * `LLMUncertaintyEstimator` reads and combines with the answer's per-token log probabilities
  * (from `AutoGGUFModel.setOutputLogProbs(true)`).
  *
  * Takes two DOCUMENT input columns, in this order: the question, and the sampled answer(s) to
  * score (one row may carry several sampled answers, e.g. from `AutoGGUFModel.setNumSamples(n)`;
  * every sample in a row is scored against that row's single question).
  *
  * The default pretrained model is `mars_token_importance`, an export of the
  * [[https://huggingface.co/duygunuryldz/MARS duygunuryldz/MARS]] checkpoint:
  * {{{
  * val marsImportance = MarsTokenImportance.pretrained()
  *   .setInputCols("question", "completions")
  *   .setOutputCol("token_importance")
  * }}}
  *
  * A self-exported ONNX checkpoint can be used instead, laid out as `<model_dir>/model.onnx` plus
  * `<model_dir>/assets/vocab.txt`, and loaded with
  * `MarsTokenImportance.loadSavedModel("<model_dir>", spark)`. It must be a
  * `BertForTokenClassification` with `num_labels=3`: `[0:2]` is a phrase-boundary class and `[2]`
  * a per-token importance score.
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
class MarsTokenImportance(override val uid: String)
    extends AnnotatorModel[MarsTokenImportance]
    with HasBatchedAnnotate[MarsTokenImportance]
    with WriteOnnxModel
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("MARS_TOKEN_IMPORTANCE"))

  /** Input Annotator Types: DOCUMENT (question), DOCUMENT (sampled answer(s))
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT)

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

  /** Maximum combined (question + answer) sequence length to process (Default: `512`); longer
    * inputs are truncated, splitting the budget evenly between question and answer
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

  private var _model: Option[Broadcast[MarsClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, onnxWrapper: OnnxWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new MarsClassification(onnxWrapper, $$(vocabulary), getCaseSensitive)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: MarsClassification = _model.get.value

  /** Whether to lowercase before tokenizing (Default: `false`, matching the public
    * `bert-base-uncased`-derived MARS checkpoint, which already lowercases internally)
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  setDefault(batchSize -> 8, maxSentenceLength -> 512, caseSensitive -> false)

  private def phrasesToJson(phrases: Array[MarsPhrase]): String = {
    compact(render(phrases.toSeq.map { p =>
      ("begin" -> p.begin) ~ ("end" -> p.end) ~ ("importance" -> p.importance.toDouble)
    }))
  }

  /** Scores every sampled answer in each row against that row's single question.
    *
    * @param batchedAnnotations
    *   one `Array[Annotation]` per row: the question annotation followed by one or more answer
    *   annotations (in input-column order, per `HasBatchedAnnotate`'s row layout)
    * @return
    *   one output annotation per input answer annotation (preserving its text and position), with
    *   `token_importance` added to its metadata
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map { rowAnnotations =>
      if (rowAnnotations.length < 2) Seq.empty[Annotation]
      else {
        val question = rowAnnotations.head.result
        val answers = rowAnnotations.tail
        answers.map { answerAnnotation =>
          val answer = answerAnnotation.result
          val phrases =
            if (answer.isEmpty) Array.empty[MarsPhrase]
            else getModelIfNotSet.tag(question, answer, getMaxSentenceLength)
          new Annotation(
            outputAnnotatorType,
            answerAnnotation.begin,
            answerAnnotation.end,
            answer,
            answerAnnotation.metadata + ("token_importance" -> phrasesToJson(phrases)))
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
      "_mars_token_importance",
      MarsTokenImportance.onnxFile)
  }
}

trait ReadablePretrainedMarsTokenImportance
    extends ParamsAndFeaturesReadable[MarsTokenImportance]
    with HasPretrained[MarsTokenImportance] {
  override val defaultModelName: Some[String] = Some("mars_token_importance")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): MarsTokenImportance = super.pretrained()

  override def pretrained(name: String): MarsTokenImportance = super.pretrained(name)

  override def pretrained(name: String, lang: String): MarsTokenImportance =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MarsTokenImportance =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMarsTokenImportanceModel extends ReadOnnxModel {
  this: ParamsAndFeaturesReadable[MarsTokenImportance] =>

  override val onnxFile: String = "mars_token_importance_onnx"

  def readModel(instance: MarsTokenImportance, path: String, spark: SparkSession): Unit = {
    val onnxWrapper = readOnnxModel(path, spark, "mars_token_importance_onnx")
    instance.setModelIfNotSet(spark, onnxWrapper)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): MarsTokenImportance = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    require(
      detectedEngine == ONNX.name,
      s"MarsTokenImportance only supports ONNX models. $notSupportedEngineError")

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val onnxWrapper =
      OnnxModelStaging.readWithDistinctName(spark, localModelPath, "mars_token_importance")

    new MarsTokenImportance()
      .setVocabulary(vocabs)
      .setModelIfNotSet(spark, onnxWrapper)
  }
}

/** This is the companion object of [[MarsTokenImportance]]. Please refer to that class for the
  * documentation.
  */
object MarsTokenImportance
    extends ReadablePretrainedMarsTokenImportance
    with ReadMarsTokenImportanceModel
