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

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.Phi3
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Phi-3
  *
  * The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open
  * model trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered
  * publicly available website data, with an emphasis on high-quality and reasoning-dense
  * properties. The model belongs to the Phi-3 family with the Mini version in two variants 4K and
  * 128K which is the context length (in tokens) that it can support.
  *
  * After initial training, the model underwent a post-training process that involved supervised
  * fine-tuning and direct preference optimization to enhance its ability to follow instructions
  * and adhere to safety measures. When evaluated against benchmarks that test common sense,
  * language understanding, mathematics, coding, long-term context, and logical reasoning, the
  * Phi-3 Mini-128K-Instruct demonstrated robust and state-of-the-art performance among models
  * with fewer than 13 billion parameters.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val phi3 = Phi3Transformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"phi_3_mini_128k_instruct"`, if no name is provided. For available
  * pretrained models please see the [[https://sparknlp.org/models?q=phi3 Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/Phi3TestSpec.scala Phi3TestSpec]].
  *
  * '''References:'''
  *   - [[https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/]]
  *   - [[https://arxiv.org/abs/2404.14219]]
  *
  * '''Paper Abstract:'''
  *
  * ''We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion
  * tokens, whose overall performance, as measured by both academic benchmarks and internal
  * testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69%
  * on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. The
  * innovation lies entirely in our dataset for training, a scaled-up version of the one used for
  * phi-2, composed of heavily filtered publicly available web data and synthetic data. The model
  * is also further aligned for robustness, safety, and chat format. We also provide some initial
  * parameter-scaling results with a 7B and 14B models trained for 4.8T tokens, called phi-3-small
  * and phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75% and
  * 78% on MMLU, and 8.7 and 8.9 on MT-bench). Moreover, we also introduce phi-3-vision, a 4.2
  * billion parameter model based on phi-3-mini with strong reasoning capabilities for image and
  * text prompts. ''
  *
  * '''Note:'''
  *
  * This is a very computationally expensive module especially on larger sequence. The use of an
  * accelerator such as GPU is recommended.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.seq2seq.Phi3Transformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val phi3 = Phi3Transformer.pretrained("phi_3_mini_128k_instruct")
  *   .setInputCols(Array("documents"))
  *   .setMinOutputLength(10)
  *   .setMaxOutputLength(50)
  *   .setDoSample(false)
  *   .setTopK(50)
  *   .setNoRepeatNgramSize(3)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, phi3))
  *
  * val data = Seq(
  *   "My name is Leonardo."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                              |
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[ My name is Leonardo. I am a man of letters. I have been a man for many years. I was born in the year 1776. I came to the United States in 1776, and I have lived in the United Kingdom since 1776]|
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * }}}
  *
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
class Phi3Transformer(override val uid: String)
    extends AnnotatorModel[Phi3Transformer]
    with HasBatchedAnnotate[Phi3Transformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("Phi3TRANSFORMER"))

  /** Input annotator type : DOCUMENT
    *
    * @group param
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
    *
    * @group param
    */
  override val outputAnnotatorType: String = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): Phi3Transformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** A list of token ids which are ignored in the decoder's output (Default: `Array()`)
    *
    * @group param
    */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): Phi3Transformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  private var _model: Option[Broadcast[Phi3]] = None

  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[OpenvinoWrapper],
      spp: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Phi3(
            onnxWrappers,
            openvinoWrapper,
            spp = spp,
            generationConfig = getGenerationConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: Phi3 = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 0.7,
    topK -> 500,
    topP -> 0.9,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 4096,
    stopTokenIds -> Array())

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
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
        minOutputLength = $(minOutputLength),
        maxOutputLength = $(maxOutputLength),
        doSample = $(doSample),
        temperature = $(temperature),
        topK = $(topK),
        topP = $(topP),
        repetitionPenalty = $(repetitionPenalty),
        noRepeatNgramSize = $(noRepeatNgramSize),
        randomSeed = this.randomSeed,
        ignoreTokenIds = $(ignoreTokenIds),
        beamSize = $(beamSize),
        maxInputLength = $(maxInputLength),
        stopTokenIds = $(stopTokenIds))
    } else {
      Seq()
    }
    Seq(processedAnnotations)
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.get.decoder, "decoder_model.onnx")),
          Phi3Transformer.suffix)
        val obj = getModelIfNotSet
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          Phi3Transformer.suffix,
          Phi3Transformer.sppFile)
      case Openvino.name =>
        val wrappers = getModelIfNotSet.openvinoWrapper
        writeOpenvinoModel(
          path,
          spark,
          wrappers.get,
          Phi3Transformer.suffix,
          Phi3Transformer.openvinoFile)
        val obj = getModelIfNotSet
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          Phi3Transformer.suffix,
          Phi3Transformer.sppFile)
    }
  }
}

trait ReadablePretrainedPhi3TransformerModel
    extends ParamsAndFeaturesReadable[Phi3Transformer]
    with HasPretrained[Phi3Transformer] {
  override val defaultModelName: Some[String] = Some("phi_3_mini_128k_instruct")

  /** Java compliant-overrides */
  override def pretrained(): Phi3Transformer = super.pretrained()

  override def pretrained(name: String): Phi3Transformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): Phi3Transformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): Phi3Transformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadPhi3TransformerDLModel
    extends ReadOnnxModel
    with ReadOpenvinoModel
    with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[Phi3Transformer] =>

  override val onnxFile: String = "phi3_onnx"
  val suffix: String = "phi3"
  override val sppFile: String = "phi3_spp"
  override val openvinoFile: String = "phi3_openvino"

  def readModel(instance: Phi3Transformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val wrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val onnxWrappers =
          DecoderWrappers(decoder = wrappers("decoder_model.onnx"))
        val spp = readSentencePieceModel(path, spark, "_phi3_spp", sppFile)
        instance.setModelIfNotSet(spark, Some(onnxWrappers), None, spp)
      case Openvino.name =>
        val ovWrapper =
          readOpenvinoModel(path, spark, "_phi3_ov")
        val spp = readSentencePieceModel(path, spark, "_phi3_spp", sppFile)
        instance.setModelIfNotSet(spark, None, Some(ovWrapper), spp)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): Phi3Transformer = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4
    val (localModelPath, detectedEngine) =
      modelSanityCheck(modelPath, isDecoder = true)
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val beginSuppressTokens: Array[Int] =
      (modelConfig \ "begin_suppress_tokens").extract[Array[Int]]

    val suppressTokenIds: Array[Int] =
      (modelConfig \ "suppress_tokens").extract[Array[Int]]

    val forcedDecoderIds: Array[(Int, Int)] =
      (modelConfig \ "forced_decoder_ids").extract[Array[Array[Int]]].map {
        case idxWithTokenId: Array[Int] if idxWithTokenId.length == 2 =>
          (idxWithTokenId(0), idxWithTokenId(1))
        case _ =>
          throw new Exception(
            "Could not extract forced_decoder_ids. Should be a list of tuples with 2 entries.")
      }

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val bosTokenId = (modelConfig \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val vocabSize = (modelConfig \ "vocab_size").extract[Int]

    val annotatorModel = new Phi3Transformer()
      .setGenerationConfig(
        GenerationConfig(
          bosTokenId,
          padTokenId,
          eosTokenId,
          vocabSize,
          arrayOrNone(beginSuppressTokens),
          arrayOrNone(suppressTokenIds),
          arrayOrNone(forcedDecoderIds)))
    val spModel = loadSentencePieceAsset(localModelPath, "tokenizer.model")

    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    modelEngine match {
      case ONNX.name =>
        val onnxWrapperDecoder =
          OnnxWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model",
            dataFileSuffix = Some(".onnx_data"),
            onnxFileSuffix = Some(suffix))

        val onnxWrappers = DecoderWrappers(onnxWrapperDecoder)

        annotatorModel
          .setModelIfNotSet(spark, Some(onnxWrappers), None, spModel)

      case Openvino.name =>
        val openvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel.setModelIfNotSet(spark, None, Some(openvinoWrapper), spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

object Phi3Transformer
    extends ReadablePretrainedPhi3TransformerModel
    with ReadPhi3TransformerDLModel
