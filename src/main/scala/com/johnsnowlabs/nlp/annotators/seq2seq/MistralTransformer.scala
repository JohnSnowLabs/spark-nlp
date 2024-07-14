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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.Mistral
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

/** Mistral 7B
  *
  * Mistral 7B, a 7.3 billion-parameter model that stands out for its efficient and effective
  * performance in natural language processing. Surpassing Llama 2 13B across all benchmarks and
  * excelling over Llama 1 34B in various aspects, Mistral 7B strikes a balance between English
  * language tasks and code comprehension, rivaling the capabilities of CodeLlama 7B in the
  * latter.
  *
  * Mistral 7B introduces Grouped-query attention (GQA) for quicker inference, enhancing
  * processing speed without compromising accuracy. This streamlined approach ensures a smoother
  * user experience, making Mistral 7B a practical choice for real-world applications.
  *
  * Additionally, Mistral 7B adopts Sliding Window Attention (SWA) to efficiently handle longer
  * sequences at a reduced computational cost. This feature enhances the model's ability to
  * process extensive textual input, expanding its utility in handling more complex tasks.
  *
  * In summary, Mistral 7B represents a notable advancement in language models, offering a
  * reliable and versatile solution for various natural language processing challenges.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val mistral = MistralTransformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"mistral_7b"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?q=mistral Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/MistralTestSpec.scala MistralTestSpec]].
  *
  * '''References:'''
  *   - [[https://mistral.ai/news/announcing-mistral_7b/ Mistral 7B]]
  *   - [[https://github.com/mistralai/mistral-src]]
  *
  * '''Paper Abstract:'''
  *
  * ''We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior
  * performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated
  * benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model
  * leverages grouped-query attention (GQA) for faster inference, coupled with sliding window
  * attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference
  * cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that
  * surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks. Our models are
  * released under the Apache 2.0 license.''
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
  * import com.johnsnowlabs.nlp.annotators.seq2seq.MistralTransformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val mistral = MistralTransformer.pretrained("mistral_7b")
  *   .setInputCols(Array("documents"))
  *   .setMinOutputLength(10)
  *   .setMaxOutputLength(50)
  *   .setDoSample(false)
  *   .setTopK(50)
  *   .setNoRepeatNgramSize(3)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, mistral))
  *
  * val data = Seq(
  *   "My name is Leonardo."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  *  +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *  |result                                                                                                                                                                                              |
  *  +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *  |[Leonardo Da Vinci invented the microscope?\n Question: Leonardo Da Vinci invented the microscope?\n Answer: No, Leonardo Da Vinci did not invent the microscope. The first microscope was invented |
  *  | in the late 16th century, long after Leonardo']                                                                                                                                                    |
  *  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *  +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
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
class MistralTransformer(override val uid: String)
    extends AnnotatorModel[MistralTransformer]
    with HasBatchedAnnotate[MistralTransformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("MistralTRANSFORMER"))

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
  def setRandomSeed(value: Int): MistralTransformer.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): MistralTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  private var _model: Option[Broadcast[Mistral]] = None

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
          new Mistral(
            onnxWrappers,
            openvinoWrapper,
            spp = spp,
            generationConfig = getGenerationConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: Mistral = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 200,
    doSample -> false,
    temperature -> 1,
    topK -> 50,
    topP -> 1,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 4096)

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
        maxInputLength = $(maxInputLength))
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
          MistralTransformer.suffix)
        val obj = getModelIfNotSet
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          MistralTransformer.suffix,
          MistralTransformer.sppFile)
      case Openvino.name =>
        val wrappers = getModelIfNotSet.openvinoWrapper
        writeOpenvinoModel(
          path,
          spark,
          wrappers.get,
          MistralTransformer.suffix,
          MistralTransformer.openvinoFile)
        val obj = getModelIfNotSet
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          MistralTransformer.suffix,
          MistralTransformer.sppFile)
    }
  }
}

trait ReadablePretrainedMistralTransformerModel
    extends ParamsAndFeaturesReadable[MistralTransformer]
    with HasPretrained[MistralTransformer] {
  override val defaultModelName: Some[String] = Some("mistral_7b")

  /** Java compliant-overrides */
  override def pretrained(): MistralTransformer = super.pretrained()

  override def pretrained(name: String): MistralTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): MistralTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MistralTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMistralTransformerDLModel
    extends ReadOnnxModel
    with ReadOpenvinoModel
    with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[MistralTransformer] =>

  override val onnxFile: String = "mistral_onnx"
  val suffix: String = "_mistral"
  override val sppFile: String = "mistral_spp"
  override val openvinoFile: String = "mistral_openvino"

  def readModel(instance: MistralTransformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val wrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val onnxWrappers =
          DecoderWrappers(decoder = wrappers("decoder_model.onnx"))
        val spp = readSentencePieceModel(path, spark, "_mistral_spp", sppFile)
        instance.setModelIfNotSet(spark, Some(onnxWrappers), None, spp)
      case Openvino.name =>
        val ovWrapper =
          readOpenvinoModel(path, spark, "_mistral_ov")
        val spp = readSentencePieceModel(path, spark, "_mistral_spp", sppFile)
        instance.setModelIfNotSet(spark, None, Some(ovWrapper), spp)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): MistralTransformer = {
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

    val annotatorModel = new MistralTransformer()
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

object MistralTransformer
    extends ReadablePretrainedMistralTransformerModel
    with ReadMistralTransformerDLModel
