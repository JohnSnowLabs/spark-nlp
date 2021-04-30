/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import com.johnsnowlabs.ml.tensorflow.sentencepiece.{ReadSentencePieceModel, SentencePieceWrapper, WriteSentencePieceModel}
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowT5, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, HasSimpleAnnotate, ParamsAndFeaturesReadable, ParamsAndFeaturesWritable}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.io.File

class T5Transformer(override val uid: String)
  extends AnnotatorModel[T5Transformer]
    with HasSimpleAnnotate[T5Transformer]
    with ParamsAndFeaturesWritable
    with WriteTensorflowModel
    with WriteSentencePieceModel {

  def this() = this(Identifiable.randomUID("T5TRANSFORMER"))

  /** Output annotator type : TOKEN
   *
   * @group param
   * */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
   *
   * @group param
   * */
  override val outputAnnotatorType: String = DOCUMENT

  /**
   * Set transformer task, e.g. 'summarize'
   *
   * @group param
   */
  val task = new Param[String](this, "task", "Set transformer task, e.g. 'summarize'")

  /** @group setParam * */
  def setTask(value: String): T5Transformer.this.type = {
    if (get(task).isEmpty)
      set(task, value)
    this
  }

  /**
   * Minimum length of the sequence to be generated
   *
   * @group param
   */
  val minOutputLength = new IntParam(this, "minOutputLength", "Minimum length of the sequence to be generated")

  /** @group setParam * */
  def setMinOutputLength(value: Int): T5Transformer.this.type = {
    set(minOutputLength, value)
    this
  }

  /** @group getParam * */
  def getMinOutputLength: Int = $(this.minOutputLength)

  /**
   * Maximum length of the sequence to be generated
   *
   * @group param
   */
  val maxOutputLength = new IntParam(this, "maxOutputLength", "Maximum length of the sequence to be generated")

  /** @group setParam * */
  def setMaxOutputLength(value: Int): T5Transformer.this.type = {
    set(maxOutputLength, value)
    this
  }

  /** @group getParam * */
  def getMaxOutputLength: Int = $(this.maxOutputLength)

  /**
   * Whether or not to use sampling; use greedy decoding otherwise
   *
   * @group param
   */
  val doSample = new BooleanParam(this, "doSample", "Whether or not to use sampling; use greedy decoding otherwise")

  /** @group setParam * */
  def setDoSample(value: Boolean): T5Transformer.this.type = {
    set(doSample, value)
    this
  }

  /** @group getParam * */
  def getDoSample: Boolean = $(this.doSample)

  /**
   * The value used to module the next token probabilities
   *
   * @group param
   */
  val temperature = new DoubleParam(this, "temperature", "The value used to module the next token probabilities")

  /** @group setParam * */
  def setTemperature(value: Double): T5Transformer.this.type = {
    set(temperature, value)
    this
  }

  /** @group getParam * */
  def getTemperature: Double = $(this.temperature)

  /**
   * The number of highest probability vocabulary tokens to keep for top-k-filtering
   *
   * @group param
   */
  val topK = new IntParam(this, "topK", "The number of highest probability vocabulary tokens to keep for top-k-filtering")

  /** @group setParam * */
  def setTopK(value: Int): T5Transformer.this.type = {
    set(topK, value)
    this
  }

  /** @group getParam * */
  def getTopK: Int = $(this.topK)

  /**
   * If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation
   *
   * @group param
   */
  val topP = new DoubleParam(this, "topP", "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation")

  /** @group setParam * */
  def setTopP(value: Double): T5Transformer.this.type = {
    set(topP, value)
    this
  }

  /** @group getParam * */
  def getTopP: Double = $(this.topP)

  /**
   * The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>` for more details
   *
   * @group param
   */
  val repetitionPenalty = new DoubleParam(this, "repetitionPenalty", "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details")

  /** @group setParam * */
  def setRepetitionPenalty(value: Double): T5Transformer.this.type = {
    set(repetitionPenalty, value)
    this
  }

  /** @group getParam * */
  def getRepetitionPenalty: Double = $(this.repetitionPenalty)

  /**
   * If set to int > 0, all ngrams of that size can only occur once
   *
   * @group param
   */
  val noRepeatNgramSize = new IntParam(this, "noRepeatNgramSize", "If set to int > 0, all ngrams of that size can only occur once")

  /** @group setParam * */
  def setNoRepeatNgramSize(value: Int): T5Transformer.this.type = {
    set(noRepeatNgramSize, value)
    this
  }

  /** @group getParam */
  def getNoRepeatNgramSize: Int = $(this.noRepeatNgramSize)

  /**
   * @group param
   */
  var randomSeed: Option[Long] = None

  /** @group setParam */
  def setRandomSeed(value: Long): T5Transformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** @group getParam */
  def getRandomSeed: Option[Long] = this.randomSeed

  /**
   * ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): T5Transformer.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  private var _tfModel: Option[Broadcast[TensorflowT5]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tfWrapper: TensorflowWrapper, spp: SentencePieceWrapper): this.type = {
    if (_tfModel.isEmpty) {
      _tfModel = Some(
        spark.sparkContext.broadcast(
          new TensorflowT5(tfWrapper, spp, configProtoBytes = getConfigProtoBytes)
        )
      )
    }
    this
  }

  /** @group getParam * */
  def getModelIfNotSet: TensorflowT5 = _tfModel.get.value

  setDefault(
    task -> "",
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0
  )


  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val nonEmptySentences = annotations.filter(_.result.nonEmpty)

    if (nonEmptySentences.nonEmpty) {
      this.getModelIfNotSet.generateSeq2Seq(
        sentences = nonEmptySentences,
        batchSize = 1,
        minOutputLength = $(minOutputLength),
        maxOutputLength = $(maxOutputLength),
        doSample = $(doSample),
        temperature = $(temperature),
        topK = $(topK),
        topP = $(topP),
        repetitionPenalty = $(repetitionPenalty),
        noRepeatNgramSize = $(noRepeatNgramSize),
        task = $(task),
        randomSeed = this.randomSeed
      )
    } else {
      Seq.empty[Annotation]
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_t5", T5Transformer.tfFile, configProtoBytes = getConfigProtoBytes)
    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, "_t5", T5Transformer.sppFile)

  }
}

trait ReadablePretrainedT5TransformerModel extends ParamsAndFeaturesReadable[T5Transformer] with HasPretrained[T5Transformer] {
  override val defaultModelName: Some[String] = Some("t5_small")

  /** Java compliant-overrides */
  override def pretrained(): T5Transformer = super.pretrained()

  override def pretrained(name: String): T5Transformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): T5Transformer = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): T5Transformer = super.pretrained(name, lang, remoteLoc)
}

trait ReadT5TransformerTensorflowModel extends ReadTensorflowModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[T5Transformer] =>

  override val tfFile: String = "t5_tensorflow"
  override val sppFile: String = "t5_spp"

  def readTensorflow(instance: T5Transformer, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_t5_tf")
    val spp = readSentencePieceModel(path, spark, "_t5_spp", sppFile)
    instance.setModelIfNotSet(spark, tf, spp)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): T5Transformer = {

    val f = new File(folder)
    val sppModelPath = folder + "/assets"
    val savedModel = new File(folder, "saved_model.pb")
    val sppModel = new File(sppModelPath, "spiece.model")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )
    require(sppModel.exists(), s"SentencePiece model not found in folder $sppModelPath")

    val (wrapper, _) = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"))
    val spp = SentencePieceWrapper.read(sppModel.toString)

    val t5model = new T5Transformer().setModelIfNotSet(spark, wrapper, spp)

    t5model
  }
}


object T5Transformer extends ReadablePretrainedT5TransformerModel with ReadT5TransformerTensorflowModel with ReadSentencePieceModel