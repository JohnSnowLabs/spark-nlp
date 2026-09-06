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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.ai.CrossEncoderClassification
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
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** CrossEncoder brings cross-encoder relevance scoring (as in `sentence-transformers`
  * `CrossEncoder`) into Spark NLP as a first-class annotator.
  *
  * It takes two row-aligned document columns, jointly encodes each row's pair as a single
  * sequence `[CLS] text_a [SEP] text_b [SEP]`, runs one forward pass through a BERT-family
  * transformer with a single-logit regression head, and writes one score per row to a single
  * output column. The logit is squashed with a sigmoid, so every score lands in `[0, 1]`. Row `i`
  * of the first column and row `i` of the second column produce row `i` of the output
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val crossEncoder = CrossEncoder.pretrained()
  *   .setInputCols("document1", "document2")
  *   .setOutputCol("score")
  * }}}
  * The default model is `"cross_encoder_ms_marco_minilm_l6_v2"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Text+Classification Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val document = new MultiDocumentAssembler()
  *   .setInputCols("query", "passage")
  *   .setOutputCols("document1", "document2")
  *
  * val crossEncoder = CrossEncoder.pretrained()
  *   .setInputCols("document1", "document2")
  *   .setOutputCol("score")
  *
  * val pipeline = new Pipeline().setStages(Array(document, crossEncoder))
  *
  * val data = Seq(
  *   ("How many people live in Berlin?", "Berlin is well known for its museums."))
  *   .toDF("query", "passage")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("score.result").show(false)
  * }}}
  *
  * @see
  *   [[BertForSequenceClassification]] for single-sequence classification
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based classifiers
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
class CrossEncoder(override val uid: String)
    extends AnnotatorModel[CrossEncoder]
    with HasBatchedAnnotate[CrossEncoder]
    with WriteOnnxModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("CrossEncoder"))

  /** Input Annotator Types: DOCUMENT, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT)

  /** Output Annotator Types: CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CATEGORY

  def sentenceStartTokenId: Int = $$(vocabulary)("[CLS]")

  def sentenceEndTokenId: Int = $$(vocabulary)("[SEP]")

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  private var _model: Option[Broadcast[CrossEncoderClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, onnxWrapper: OnnxWrapper): CrossEncoder = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new CrossEncoderClassification(
            onnxWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: CrossEncoderClassification = _model.get.value

  /** Whether to lowercase tokens or not (Default: `false`).
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  setDefault(batchSize -> 8, caseSensitive -> false)

  /** Takes a batch of row-aligned document pairs and produces one CATEGORY score per row.
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols (document1, document2) generated by
    *   previous annotators
    * @return
    *   exactly one score Annotation per input row, in input order
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val indexedPairs = batchedAnnotations.zipWithIndex.map { case (annotations, i) =>
      val documents = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)
      val pair = if (documents.length >= 2) Some((documents.head, documents(1))) else None
      (i, pair)
    }

    val presentPairs = indexedPairs.collect { case (i, Some(pair)) => (i, pair) }

    if (presentPairs.isEmpty) return batchedAnnotations.map(_ => Seq.empty[Annotation])

    val scores =
      getModelIfNotSet.predictScore(presentPairs.map(_._2), $(batchSize), $(caseSensitive))

    val scoreByIndex = presentPairs.map(_._1).zip(scores).toMap

    batchedAnnotations.indices.map { i =>
      scoreByIndex.get(i).map(Seq(_)).getOrElse(Seq.empty[Annotation])
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeOnnxModel(
      path,
      spark,
      getModelIfNotSet.onnxWrapper,
      "_cross_encoder_classification",
      CrossEncoder.onnxFile)
  }

}

trait ReadablePretrainedCrossEncoderModel
    extends ParamsAndFeaturesReadable[CrossEncoder]
    with HasPretrained[CrossEncoder] {
  override val defaultModelName: Some[String] = Some("cross_encoder_ms_marco_minilm_l6_v2")

  /** Java compliant-overrides */
  override def pretrained(): CrossEncoder = super.pretrained()

  override def pretrained(name: String): CrossEncoder =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): CrossEncoder =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): CrossEncoder =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadCrossEncoderDLModel extends ReadOnnxModel {
  this: ParamsAndFeaturesReadable[CrossEncoder] =>

  override val onnxFile: String = "cross_encoder_classification_onnx"

  def readModel(instance: CrossEncoder, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper = readOnnxModel(path, spark, "cross_encoder_classification_onnx")
        instance.setModelIfNotSet(spark, onnxWrapper)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): CrossEncoder = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val annotatorModel = new CrossEncoder()
      .setVocabulary(vocabs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel.setModelIfNotSet(spark, onnxWrapper)

      case _ =>
        throw new Exception(
          "CrossEncoder currently only supports ONNX models. " +
            notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[CrossEncoder]]. Please refer to that class for the
  * documentation.
  */
object CrossEncoder extends ReadablePretrainedCrossEncoderModel with ReadCrossEncoderDLModel
