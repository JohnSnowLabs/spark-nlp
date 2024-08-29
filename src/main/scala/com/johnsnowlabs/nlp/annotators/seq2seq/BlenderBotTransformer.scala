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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.ai.BlenderBotGenerate
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{loadTextAsset, modelSanityCheck, notSupportedEngineError}
import com.johnsnowlabs.ml.util.TensorFlow
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

class BlenderBotTransformer(override val uid: String)
  extends AnnotatorModel[BlenderBotTransformer]
  with HasBatchedAnnotate[BlenderBotTransformer]
  with ParamsAndFeaturesWritable
  with WriteTensorflowModel
  with HasEngine
  with HasGeneratorProperties {

  def this() = this(Identifiable.randomUID("BlenderBotTransformer"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
   * type
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  //TODO: Can we move signatures, configProtoBytes to a Trait??

  /** It contains TF model signatures for the laded saved model
   *
   * @group param
   */
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
   * config_proto.SerializeToString()
   *
   * @group param
   */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): BlenderBotTransformer.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  //TODO: Can we add vocabulary and merges to a Trait??

  /** Vocabulary used to encode the words to ids with bpeTokenizer.encode
   *
   * @group param
   */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Holding merges.txt coming from RoBERTa model
   *
   * @group param
   */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges").setProtected()

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  /** Cache internal state of the model to improve performance
   *
   * @group param
   */
  val useCache =
    new BooleanParam(parent = this, name = "useCache", doc = "Cache internal state of the model")

  protected def setUseCache(value: Boolean): BlenderBotTransformer.this.type = {
    set(useCache, value)
    this
  }

  def getUseCache: Boolean = $(useCache)

  /** A list of token ids which are ignored in the decoder 's output (Default: `Array()`)
   *
   * @group param
   */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): BlenderBotTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  private var _tfModel: Option[Broadcast[BlenderBotGenerate]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tfWrapper: TensorflowWrapper,
      useCache: Boolean): this.type = {
    if (_tfModel.isEmpty) {
      setUseCache(useCache)
      _tfModel = Some(
        spark.sparkContext.broadcast(
          new BlenderBotGenerate(
            tfWrapper,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            $$(merges),
            $$(vocabulary),
            useCache = useCache)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: BlenderBotGenerate = _tfModel.get.value

  setDefault(
    task -> "",
    minOutputLength -> 0,
    maxOutputLength -> 20,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    doSample -> false,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 4,
    maxInputLength -> 512,
    useCache -> true)


  /** takes a document and annotations and produces new annotations of this annotator's annotation
   * type
   *
   * @param batchedAnnotations
   * Annotations in batches that correspond to inputAnnotationCols generated by previous
   * annotators if any
   * @return
   * any number of annotations processed for every batch of input annotations. Not necessary
   * one to one relationship
   *
   * IMPORTANT: !MUST! return sequences of equal lengths !! IMPORTANT: !MUST! return sentences
   * that belong to the same original row !! (challenging)
   */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val allAnnotations = batchedAnnotations
      .filter(_.nonEmpty)
      .zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }

    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        batchSize = $(batchSize),
        maxOutputLength = $(maxOutputLength),
        doSample = $(doSample),
        temperature = $(temperature),
        topK = $(topK),
        topP = $(topP),
        repetitionPenalty = $(repetitionPenalty),
        noRepeatNgramSize = $(noRepeatNgramSize),
        task = $(task),
        beamSize = $(beamSize),
        maxInputLength = $(maxInputLength)
      )
    } else {
      Seq()
    }

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = processedAnnotations
        // zip each annotation with its corresponding row index
        .zip(allAnnotations)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowAnnotations.nonEmpty)
        rowAnnotations
      else
        Seq.empty[Annotation]
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_blenderbot",
      BlenderBotTransformer.tfFile,
      configProtoBytes = getConfigProtoBytes,
      savedSignatures = getSignatures)
  }

}

trait ReadablePretrainedBotTransformerModel
  extends ParamsAndFeaturesReadable[BlenderBotTransformer]
  with HasPretrained[BlenderBotTransformer] {
  override val defaultModelName: Some[String] = Some("blenderbot")

  /** Java compliant-overrides */
  override def pretrained(): BlenderBotTransformer = super.pretrained()

  override def pretrained(name: String): BlenderBotTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): BlenderBotTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BlenderBotTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBlenderBotTransformer extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[BlenderBotTransformer] =>

  override val tfFile: String = "blenderbot_tensorflow"

  def readModel(instance: BlenderBotTransformer, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(
      path,
      spark,
      "_blenderbot_tf",
      savedSignatures = instance.getSignatures,
      initAllTables = false)
    instance.setModelIfNotSet(spark, tf, instance.getUseCache)
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useCache: Boolean = true): BlenderBotTransformer = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    val annotatorModel = new BlenderBotTransformer()
      .setVocabulary(vocabs)
      .setMerges(bytePairs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(
            localModelPath,
            zipped = false,
            useBundle = true,
            tags = Array("serve"))

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
         * setModelIfNotSet
         */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, wrapper, useCache)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel

  }

}

object BlenderBotTransformer
  extends ReadablePretrainedBotTransformerModel
  with ReadBlenderBotTransformer