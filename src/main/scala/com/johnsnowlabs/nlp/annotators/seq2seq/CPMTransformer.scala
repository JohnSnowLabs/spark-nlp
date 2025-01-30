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
import com.johnsnowlabs.ml.ai.CPM
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

/** MiniCPM: Unveiling the Potential of End-side Large Language Models
  *
  * MiniCPM is a series of edge-side large language models, with the base model, MiniCPM-2B,
  * having 2.4B non-embedding parameters. It ranks closely with Mistral-7B on comprehensive
  * benchmarks (with better performance in Chinese, mathematics, and coding abilities), surpassing
  * models like Llama2-13B, MPT-30B, and Falcon-40B. On the MTBench benchmark, which is closest to
  * user experience, MiniCPM-2B also outperforms many representative open-source models such as
  * Llama2-70B-Chat, Vicuna-33B, Mistral-7B-Instruct-v0.1, and Zephyr-7B-alpha.
  *
  * After DPO, MiniCPM outperforms Llama2-70B-Chat, Vicuna-33B, Mistral-7B-Instruct-v0.1,
  * Zephyr-7B-alpha, etc. on MTBench.
  *
  * MiniCPM-V, based on MiniCPM-2B, achieves the best overall performance among multimodel models
  * of the same scale, surpassing existing multimodal large models built on Phi-2 and achieving
  * performance comparable to or even better than 9.6B Qwen-VL-Chat on some tasks.
  *
  * MiniCPM can be deployed and infer on smartphones, and the speed of streaming output is
  * relatively higher than the verbal speed of human.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val cpm = CPMTransformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"mini_cpm_2b_8bit"`, if no name is provided. For available pretrained
  * models please see the [[https://sparknlp.org/models?q=cpm Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/CPMTestSpec.scala CPMTestSpec]].
  *
  * '''References:'''
  *   - [[https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20 MiniCPM: Unveiling the Potential of End-side Large Language Models]]
  *   - [[https://github.com/OpenBMB/MiniCPM]]
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
  * import com.johnsnowlabs.nlp.annotators.seq2seq.CPMTransformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val cpm = CPMTransformer.pretrained("mini_cpm_2b_8bit")
  *   .setInputCols(Array("documents"))
  *   .setMinOutputLength(10)
  *   .setMaxOutputLength(50)
  *   .setDoSample(false)
  *   .setTopK(50)
  *   .setNoRepeatNgramSize(3)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, cpm))
  *
  * val data = Seq(
  *   "My name is Leonardo."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                                 |
  * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[ My name is Leonardo. I am a student at the University of California, Los Angeles. I have a passion for writing and learning about different cultures. I enjoy playing basketball and watching movies]|
  * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
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
class CPMTransformer(override val uid: String)
    extends AnnotatorModel[CPMTransformer]
    with HasBatchedAnnotate[CPMTransformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("CPMTRANSFORMER"))

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
  def setRandomSeed(value: Int): CPMTransformer.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): CPMTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  private var _model: Option[Broadcast[CPM]] = None

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
          new CPM(
            onnxWrappers,
            openvinoWrapper,
            spp = spp,
            generationConfig = getGenerationConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: CPM = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 50,
    doSample -> true,
    temperature -> 0.8,
    topK -> 100,
    topP -> 0.8,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
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
          CPMTransformer.suffix)
        val obj = getModelIfNotSet
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          CPMTransformer.suffix,
          CPMTransformer.sppFile)
      case Openvino.name =>
        val wrappers = getModelIfNotSet.openvinoWrapper
        writeOpenvinoModel(
          path,
          spark,
          wrappers.get,
          CPMTransformer.suffix,
          CPMTransformer.openvinoFile)
        val obj = getModelIfNotSet
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          CPMTransformer.suffix,
          CPMTransformer.sppFile)
    }
  }
}

trait ReadablePretrainedCPMTransformerModel
    extends ParamsAndFeaturesReadable[CPMTransformer]
    with HasPretrained[CPMTransformer] {
  override val defaultModelName: Some[String] = Some("mini_cpm_2b_8bit")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): CPMTransformer = super.pretrained()

  override def pretrained(name: String): CPMTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): CPMTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): CPMTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadCPMTransformerDLModel
    extends ReadOnnxModel
    with ReadOpenvinoModel
    with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[CPMTransformer] =>

  override val onnxFile: String = "cpm_onnx"
  val suffix: String = "_cpm"
  override val sppFile: String = "cpm_spp"
  override val openvinoFile: String = "cpm_openvino"

  def readModel(instance: CPMTransformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val wrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val onnxWrappers =
          DecoderWrappers(decoder = wrappers("decoder_model.onnx"))
        val spp = readSentencePieceModel(path, spark, "_cpm_spp", sppFile)
        instance.setModelIfNotSet(spark, Some(onnxWrappers), None, spp)
      case Openvino.name =>
        val ovWrapper =
          readOpenvinoModel(path, spark, "_cpm_ov")
        val spp = readSentencePieceModel(path, spark, "_cpm_spp", sppFile)
        instance.setModelIfNotSet(spark, None, Some(ovWrapper), spp)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): CPMTransformer = {
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

    val annotatorModel = new CPMTransformer()
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

object CPMTransformer extends ReadablePretrainedCPMTransformerModel with ReadCPMTransformerDLModel
