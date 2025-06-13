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
import com.johnsnowlabs.ml.ai.Phi4
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadSentencePieceAsset,
  loadTextAsset,
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

/**
  * Phi-4: State-of-the-art open model by Microsoft Research
  *
  * phi-4 is a 14B parameter, dense decoder-only Transformer model trained on 9.8T tokens, designed for advanced reasoning, code, and general NLP tasks. For more details, see:
  * https://huggingface.co/microsoft/phi-4
  *
  * == Model Overview ==
  * - 14B parameters, dense decoder-only Transformer
  * - 16K context length
  * - Trained on 9.8T tokens (synthetic, public domain, academic, Q&A, code)
  * - Focus on high-quality, advanced reasoning, math, code, and general NLP
  * - Multilingual data: ~8% (primarily English)
  * - Released under MIT License
  *
  * == Intended Use ==
  * - General-purpose AI, research, and generative features
  * - Memory/compute constrained and latency-bound environments
  * - Reasoning, logic, and code generation
  *
  * == Benchmarks ==
  * - MMLU: 84.8 | HumanEval: 82.6 | GPQA: 56.1 | DROP: 75.5 | MATH: 80.6
  * - Outperforms or matches other 14B/70B models on many tasks
  *
  * == Safety & Limitations ==
  * - Safety alignment via SFT and DPO, red-teamed by Microsoft AIRT
  * - Not intended for high-risk or consequential domains without further assessment
  * - Primarily English; other languages may have reduced performance
  * - May generate inaccurate, offensive, or biased content; use with care
  *
  * == Usage ==
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{ {
  * val phi4 = Phi4Transformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"phi-4"`, if no name is provided. For available pretrained models please
  * see the [[https://huggingface.co/microsoft/phi-4 Models Hub]].
  *
  * '''Note:''' This is a resource-intensive module, especially with larger models and sequences.
  * Use of accelerators such as GPUs is strongly recommended.
  *
  * '''References:'''
  *   - https://huggingface.co/microsoft/phi-4
  *   - arXiv:2412.08905
  *
  * == Example ==
  * {{ {
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val phi4 = Phi4Transformer.pretrained("phi-4")
  *   .setInputCols(Array("documents"))
  *   .setMaxOutputLength(60)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, phi4))
  *
  * val data = Seq(
  *   (1, "<|im_start|>system<|im_sep|>\nYou are a helpful assistant!\n<|im_start|>user<|im_sep|>\nWhat is Phi-4?\n<|im_start|>assistant<|im_sep|>\n")
  * ).toDF("id", "text")
  *
  * val result = pipeline.fit(data).transform(data)
  * result.select("generation.result").show(truncate = false)
  * }}}
  */
class Phi4Transformer(override val uid: String)
    extends AnnotatorModel[Phi4Transformer]
    with HasBatchedAnnotate[Phi4Transformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  def this() = this(Identifiable.randomUID("PHI4TRANSFORMER"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)
  override val outputAnnotatorType: String = DOCUMENT

  def setRandomSeed(value: Int): Phi4Transformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  def setIgnoreTokenIds(tokenIds: Array[Int]): Phi4Transformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges").setProtected()
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  val addedTokens: MapFeature[String, Int] = new MapFeature(this, "addedTokens").setProtected()
  def setAddedTokens(value: Map[String, Int]): this.type = set(addedTokens, value)

  override val stopTokenIds =
    new IntArrayParam(this, "stopTokenIds", "Stop tokens to terminate the generation")
  override def setStopTokenIds(value: Array[Int]): this.type = {
    set(stopTokenIds, value)
  }
  override def getStopTokenIds: Array[Int] = $(stopTokenIds)

  private var _model: Option[Broadcast[Phi4]] = None

  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()
  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)
  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[OpenvinoWrapper]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Phi4(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            generationConfig = getGenerationConfig)))
    }
    this
  }
  def getModelIfNotSet: Phi4 = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 0.6,
    topK -> -1,
    topP -> 0.9,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 4096,
    stopTokenIds -> Array(128001))

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
          Phi4Transformer.suffix)
      case Openvino.name =>
        val wrappers = getModelIfNotSet.openvinoWrapper
        writeOpenvinoModel(
          path,
          spark,
          wrappers.get,
          Phi4Transformer.suffix,
          Phi4Transformer.openvinoFile)
    }
  }
}

trait ReadablePretrainedPhi4TransformerModel
    extends ParamsAndFeaturesReadable[Phi4Transformer]
    with HasPretrained[Phi4Transformer] {
  override val defaultModelName: Some[String] = Some("phi-4")

  override def pretrained(): Phi4Transformer = super.pretrained()
  override def pretrained(name: String): Phi4Transformer = super.pretrained(name)
  override def pretrained(name: String, lang: String): Phi4Transformer =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): Phi4Transformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadPhi4TransformerDLModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[Phi4Transformer] =>

  override val onnxFile: String = "phi4_onnx"
  val suffix: String = "_phi4"
  override val openvinoFile: String = "phi4_openvino"

  def readModel(instance: Phi4Transformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val wrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val onnxWrappers =
          DecoderWrappers(decoder = wrappers("decoder_model.onnx"))
        instance.setModelIfNotSet(spark, Some(onnxWrappers), None)
      case Openvino.name =>
        val ovWrapper =
          readOpenvinoModel(path, spark, "_phi4_ov")
        instance.setModelIfNotSet(spark, None, Some(ovWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): Phi4Transformer = {
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

    val tokenizerPath = s"$localModelPath/assets/tokenizer.json"
    val tokenizerExists = new java.io.File(tokenizerPath).exists()
    val (vocabs, addedTokens, bytePairs) = if (tokenizerExists) {
      val tokenizerConfig: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
      var vocabs: Map[String, Int] =
        (tokenizerConfig \ "model" \ "vocab").extract[Map[String, Int]]
      val bytePairs = (tokenizerConfig \ "model" \ "merges")
        .extract[List[String]]
        .map(_.split(" "))
        .filter(w => w.length == 2)
        .map { case Array(c1, c2) => (c1, c2) }
        .zipWithIndex
        .toMap
      val addedTokens = (tokenizerConfig \ "added_tokens")
        .extract[List[Map[String, Any]]]
        .map { token =>
          val id = token("id").asInstanceOf[BigInt].intValue()
          val content = token("content").asInstanceOf[String]
          (content, id)
        }
        .toMap
      addedTokens.foreach { case (content, id) =>
        vocabs += (content -> id)
      }
      (vocabs, addedTokens, bytePairs)
    } else {
      val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
      val addedTokens = loadTextAsset(localModelPath, "added_tokens.txt").zipWithIndex.toMap
      val bytePairs = loadTextAsset(localModelPath, "merges.txt")
        .map(_.split(" "))
        .filter(w => w.length == 2)
        .map { case Array(c1, c2) => (c1, c2) }
        .zipWithIndex
        .toMap
      (vocabs, addedTokens, bytePairs)
    }
    val annotatorModel = new Phi4Transformer()
      .setGenerationConfig(
        GenerationConfig(
          bosTokenId,
          padTokenId,
          eosTokenId,
          vocabSize,
          arrayOrNone(beginSuppressTokens),
          arrayOrNone(suppressTokenIds),
          arrayOrNone(forcedDecoderIds)))
      .setVocabulary(vocabs)
      .setMerges(bytePairs)
      .setAddedTokens(addedTokens)

    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    detectedEngine match {
      case ONNX.name =>
        val onnxWrapperDecoder =
          OnnxWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model")

        val onnxWrappers = DecoderWrappers(onnxWrapperDecoder)

        annotatorModel
          .setModelIfNotSet(spark, Some(onnxWrappers), None)
      case Openvino.name =>
        val openvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel.setModelIfNotSet(spark, None, Some(openvinoWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object Phi4Transformer
    extends ReadablePretrainedPhi4TransformerModel
    with ReadPhi4TransformerDLModel
