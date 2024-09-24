/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.ml.ai.Bart
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
  * Translation, and Comprehension Transformer
  *
  * The Facebook BART (Bidirectional and Auto-Regressive Transformer) model is a state-of-the-art
  * language generation model that was introduced by Facebook AI in 2019. It is based on the
  * transformer architecture and is designed to handle a wide range of natural language processing
  * tasks such as text generation, summarization, and machine translation.
  *
  * BART is unique in that it is both bidirectional and auto-regressive, meaning that it can
  * generate text both from left-to-right and from right-to-left. This allows it to capture
  * contextual information from both past and future tokens in a sentence,resulting in more
  * accurate and natural language generation.
  *
  * The model was trained on a large corpus of text data using a combination of unsupervised and
  * supervised learning techniques. It incorporates pretraining and fine-tuning phases, where the
  * model is first trained on a large unlabeled corpus of text, and then fine-tuned on specific
  * downstream tasks.
  *
  * BART has achieved state-of-the-art performance on a wide range of NLP tasks, including
  * summarization, question-answering, and language translation. Its ability to handle multiple
  * tasks and its high performance on each of these tasks make it a versatile and valuable tool
  * for natural language processing applications.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val bart = BartTransformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"distilbart_xsum_12_6"`, if no name is provided. For available
  * pretrained models please see the [[https://sparknlp.org/models?q=bart Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/BartTestSpec.scala BartTestSpec]].
  *
  * '''References:'''
  *   - [[https://aclanthology.org/2020.acl-main.703.pdf BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension]]
  *   - [[https://github.com/pytorch/fairseq]]
  *
  * '''Paper Abstract:'''
  *
  * '' We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART
  * is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model
  * to reconstruct the original text. It uses a standard Tranformer-based neural machine
  * translation architecture which, despite its simplicity, can be seen as generalizing BERT (due
  * to the bidirectional encoder), GPT (with the left-to-right decoder), and other recent
  * pretraining schemes. We evaluate a number of noising approaches, finding the best performance
  * by both randomly shuffling the order of sentences and using a novel in-filling scheme, where
  * spans of text are replaced with a single mask token. BART is particularly effective when fine
  * tuned for text generation but also works well for comprehension tasks. It matches the
  * performance of RoBERTa on GLUE and SQuAD, and achieves new stateof-the-art results on a range
  * of abstractive dialogue, question answering, and summarization tasks, with gains of up to 3.5
  * ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine
  * translation, with only target language pretraining. We also replicate other pretraining
  * schemes within the BART framework, to understand their effect on end-task performance ''
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
  * import com.johnsnowlabs.nlp.annotators.seq2seq.GPT2Transformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val bart = BartTransformer.pretrained("distilbart_xsum_12_6")
  *   .setInputCols(Array("documents"))
  *   .setMinOutputLength(10)
  *   .setMaxOutputLength(30)
  *   .setDoSample(true)
  *   .setTopK(50)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, bart))
  *
  * val data = Seq(
  *   "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
  *   "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were " +
  *   "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +--------------------------------------------------------------+
  * |result                                                        |
  * +--------------------------------------------------------------+
  * |[Nearly 800 thousand customers were affected by the shutoffs.]|
  * +--------------------------------------------------------------+
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
class BartTransformer(override val uid: String)
    extends AnnotatorModel[BartTransformer]
    with HasBatchedAnnotate[BartTransformer]
    with ParamsAndFeaturesWritable
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine
    with HasGeneratorProperties {

  def this() = this(Identifiable.randomUID("BartTRANSFORMER"))

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
  def setConfigProtoBytes(bytes: Array[Int]): BartTransformer.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** A list of token ids which are ignored in the decoder 's output (Default: `Array()`)
    *
    * @group param
    */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): BartTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

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
  private var _tfModel: Option[Broadcast[Bart]] = None

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

  protected def setUseCache(value: Boolean): BartTransformer.this.type = {
    set(useCache, value)
    this
  }

  def getUseCache: Boolean = $(useCache)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tfWrapper: Option[TensorflowWrapper],
      onnxWrappers: Option[EncoderDecoderWithoutPastWrappers],
      useCache: Boolean): this.type = {
    if (_tfModel.isEmpty) {
      setUseCache(useCache)
      _tfModel = Some(
        spark.sparkContext.broadcast(
          new Bart(
            tfWrapper,
            onnxWrappers,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            $$(merges),
            $$(vocabulary),
            useCache = useCache)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: Bart = _tfModel.get.value

  setDefault(
    task -> "",
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 4,
    maxInputLength -> 512,
    useCache -> true)

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
        task = $(task),
        randomSeed = this.randomSeed,
        ignoreTokenIds = $(ignoreTokenIds),
        beamSize = $(beamSize),
        maxInputLength = $(maxInputLength))
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
    getEngine match {

      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          BartTransformer.suffix,
          BartTransformer.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)

      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrapper
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.get.encoder, "encoder_model.onnx")),
          BartTransformer.suffix)
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.get.decoder, "decoder_model.onnx")),
          BartTransformer.suffix)
    }
  }
}

trait ReadablePretrainedBartTransformerModel
    extends ParamsAndFeaturesReadable[BartTransformer]
    with HasPretrained[BartTransformer] {
  override val defaultModelName: Some[String] = Some("distilbart_xsum_12_6")

  /** Java compliant-overrides */
  override def pretrained(): BartTransformer = super.pretrained()

  override def pretrained(name: String): BartTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): BartTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BartTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBartTransformerDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[BartTransformer] =>

  override val tfFile: String = "bart_tensorflow"
  override val onnxFile: String = "bart_onnx"
  val suffix: String = "_bart"
  def readModel(instance: BartTransformer, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tf = readTensorflowModel(
          path,
          spark,
          "_bart_tf",
          savedSignatures = instance.getSignatures,
          initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tf), None, instance.getUseCache)

      case ONNX.name =>
        val decoderWrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val encoderWrappers =
          readOnnxModels(path, spark, Seq("encoder_model.onnx"), suffix)
        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            decoder = decoderWrappers("decoder_model.onnx"),
            encoder = encoderWrappers("encoder_model.onnx"))
        instance.setModelIfNotSet(spark, None, Some(onnxWrappers), instance.getUseCache)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useCache: Boolean = true): BartTransformer = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath, isEncoderDecoder = true)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new BartTransformer()
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
          .setModelIfNotSet(spark, Some(wrapper), None, useCache)

      case ONNX.name =>
        val onnxWrapperEncoder =
          OnnxWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "encoder_model",
            onnxFileSuffix = None)
        val onnxWrapperDecoder =
          OnnxWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model",
            onnxFileSuffix = None)

        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            encoder = onnxWrapperEncoder,
            decoder = onnxWrapperDecoder)

        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrappers), useCache)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

object BartTransformer
    extends ReadablePretrainedBartTransformerModel
    with ReadBartTransformerDLModel
