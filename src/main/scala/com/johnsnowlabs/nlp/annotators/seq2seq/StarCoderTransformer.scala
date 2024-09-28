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
import com.johnsnowlabs.ml.ai.StarCoder
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

/** StarCoder2: The Versatile Code Companion.
  *
  * StarCoder2 is a Transformer model designed specifically for code generation and understanding.
  * With 13 billion parameters, it builds upon the advancements of its predecessors and is trained
  * on a diverse dataset that includes multiple programming languages. This extensive training
  * allows StarCoder2 to support a wide array of coding tasks, from code completion to generation.
  *
  * StarCoder2 was developed to assist developers in writing and understanding code more
  * efficiently, making it a valuable tool for various software development and data science
  * tasks.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val starcoder2 = StarCoder2Transformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"StarCoder2-3B"`, if no name is provided. For available pretrained
  * models please see the [[https://sparknlp.org/models?q=StarCoder2 Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/StarCoder2TestSpec.scala StarCoder2TestSpec]].
  *
  * '''References:'''
  *   - [[https://huggingface.co/blog/starcoder StarCoder2: The Versatile Code Companion]]
  *   - [[https://github.com/bigcode-project/starcoder]]
  *
  * '''Paper Abstract:'''
  *
  * ''The BigCode project,1 an open-scientific collaboration focused on the responsible
  * development of Large Language Models for Code (Code LLMs), introduces StarCoder2. In
  * partnership with Software Heritage (SWH),2 we build The Stack v2 on top of the digital commons
  * of their source code archive. Alongside the SWH repositories spanning 619 programming
  * languages, we carefully select other high-quality data sources, such as GitHub pull requests,
  * Kaggle notebooks, and code documentation. This results in a training set that is 4× larger
  * than the first StarCoder dataset. We train StarCoder2 models with 3B, 7B, and 15B parameters
  * on 3.3 to 4.3 trillion tokens and thoroughly evaluate them on a comprehensive set of Code LLM
  * benchmarks.''
  *
  * '' We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size on
  * most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder2- 15B,
  * significantly outperforms other models of comparable size. In addition, it matches or
  * outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder- 33B is
  * the best-performing model at code completion for high-resource languages, we find that
  * StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as several
  * low-resource languages. We make the model weights available under an OpenRAIL license and
  * ensure full transparency regarding the training data by releasing the SoftWare Heritage
  * persistent IDentifiers (SWHIDs) of the source code data.''
  *
  * '''Note:'''
  *
  * This is a computationally intensive module, especially for larger code sequences. The use of
  * an accelerator such as GPU is recommended.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.seq2seq.StarCoder2Transformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val starcoder2 = StarCoder2Transformer.pretrained("starcoder2")
  *   .setInputCols(Array("documents"))
  *   .setMinOutputLength(10)
  *   .setMaxOutputLength(50)
  *   .setDoSample(false)
  *   .setTopK(50)
  *   .setNoRepeatNgramSize(3)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, starcoder2))
  *
  * val data = Seq(
  *   "def add(a, b):"
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                              |
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[def add(a, b): return a + b]                                                                                                                                                                       |
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
class StarCoderTransformer(override val uid: String)
    extends AnnotatorModel[StarCoderTransformer]
    with HasBatchedAnnotate[StarCoderTransformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  def this() = this(Identifiable.randomUID("StarCoderTRANSFORMER"))

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
  def setRandomSeed(value: Int): StarCoderTransformer.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): StarCoderTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

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

  private var _model: Option[Broadcast[StarCoder]] = None

  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[OpenvinoWrapper]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new StarCoder(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            generationConfig = getGenerationConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: StarCoder = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 0.6,
    topK -> 50,
    topP -> 0.9,
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
          StarCoderTransformer.suffix)
      case Openvino.name =>
        val wrappers = getModelIfNotSet.openvinoWrapper
        writeOpenvinoModel(
          path,
          spark,
          wrappers.get,
          StarCoderTransformer.suffix,
          StarCoderTransformer.openvinoFile)
    }
  }
}

trait ReadablePretrainedStarCoderTransformerModel
    extends ParamsAndFeaturesReadable[StarCoderTransformer]
    with HasPretrained[StarCoderTransformer] {
  override val defaultModelName: Some[String] = Some("starcoder")

  /** Java compliant-overrides */
  override def pretrained(): StarCoderTransformer = super.pretrained()

  override def pretrained(name: String): StarCoderTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): StarCoderTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): StarCoderTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadStarCoderTransformerDLModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[StarCoderTransformer] =>

  override val onnxFile: String = "starcoder_onnx"
  val suffix: String = "_starcoder"
  override val openvinoFile: String = "starcoder_openvino"

  def readModel(instance: StarCoderTransformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val wrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val onnxWrappers =
          DecoderWrappers(decoder = wrappers("decoder_model.onnx"))
        instance.setModelIfNotSet(spark, Some(onnxWrappers), None)
      case Openvino.name =>
        val ovWrapper =
          readOpenvinoModel(path, spark, "_starcoder_ov")
        instance.setModelIfNotSet(spark, None, Some(ovWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): StarCoderTransformer = {
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

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    val annotatorModel = new StarCoderTransformer()
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

object StarCoderTransformer
    extends ReadablePretrainedStarCoderTransformerModel
    with ReadStarCoderTransformerDLModel
