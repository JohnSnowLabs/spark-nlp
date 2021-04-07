package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.tensorflow.sentencepiece.{ReadSentencePieceModel, SentencePieceWrapper, WriteSentencePieceModel}
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowT5, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, HasSimpleAnnotate, ParamsAndFeaturesReadable, ParamsAndFeaturesWritable}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntArrayParam, IntParam, LongParam, Param}
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
    * @group anno
    * */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    * */
  override val outputAnnotatorType: String = DOCUMENT

  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  val task = new Param[String](this, "task", "Set transformer task, e.g. 'summarize'")
  val minOutputLength = new IntParam(this, "minOutputLength", "Minimum length of the sequence to be generated")
  val maxOutputLength = new IntParam(this, "maxOutputLength", "Maximum length of output text")
  val doSample = new BooleanParam(this, "doSample", "Whether or not to use sampling; use greedy decoding otherwise")
  val temperature = new DoubleParam(this, "temperature", "The value used to module the next token probabilities")
  val topK = new IntParam(this, "topK", "The number of highest probability vocabulary tokens to keep for top-k-filtering")
  val topP = new DoubleParam(this, "topP", "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation")
  val repetitionPenalty = new DoubleParam(this, "repetitionPenalty", "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details")
  val noRepeatNgramSize = new IntParam(this, "noRepeatNgramSize", "If set to int > 0, all ngrams of that size can only occur once")
  var randomSeed: Option[Long] = None
  private var _tfModel: Option[Broadcast[TensorflowT5]] = None

  def setConfigProtoBytes(bytes: Array[Int]): T5Transformer  .this.type = set(this.configProtoBytes, bytes)

  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** @group setParam **/
  def setTask(value: String): T5Transformer.this.type = {
    if (get(task).isEmpty)
      set(task, value)
    this
  }

  /** @group setParam **/
  def setMinOutputLength(value: Int): T5Transformer.this.type = {
    set(minOutputLength, value)
    this
  }

  /** @group getParam **/
  def getMinOutputLength: Int = $(this.minOutputLength)

  /** @group setParam **/
  def setMaxOutputLength(value: Int): T5Transformer.this.type = {
    set(maxOutputLength, value)
    this
  }

  /** @group getParam **/
  def getMaxOutputLength: Int = $(this.maxOutputLength)

  /** @group setParam **/
  def setDoSample(value: Boolean): T5Transformer.this.type = {
    set(doSample, value)
    this
  }

  /** @group getParam **/
  def getDoSample: Boolean = $(this.doSample)

  /** @group setParam **/
  def setTemperature(value: Double): T5Transformer.this.type = {
    set(temperature, value)
    this
  }

  /** @group getParam **/
  def getTemperature: Double = $(this.temperature)

  /** @group setParam **/
  def setTopK(value: Int): T5Transformer.this.type = {
    set(topK, value)
    this
  }

  /** @group getParam **/
  def getTopK: Int = $(this.topK)

  /** @group setParam **/
  def setTopP(value: Double): T5Transformer.this.type = {
    set(topP, value)
    this
  }

  /** @group getParam **/
  def getTopP: Double = $(this.topP)

  /** @group setParam **/
  def setRepetitionPenalty(value: Double): T5Transformer.this.type = {
    set(repetitionPenalty, value)
    this
  }

  /** @group getParam **/
  def getRepetitionPenalty: Double = $(this.repetitionPenalty)

  /** @group setParam **/
  def setNoRepeatNgramSize(value: Int): T5Transformer.this.type = {
    set(noRepeatNgramSize, value)
    this
  }

  /** @group getParam **/
  def getNoRepeatNgramSize: Int = $(this.noRepeatNgramSize)

  /** @group setParam **/
  def setRandomSeed(value: Long): T5Transformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** @group getParam **/
  def getRandomSeed: Option[Long] = this.randomSeed

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

    //        randomSeed = $(randomSeed),
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
    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, "_t5",  T5Transformer.sppFile)

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
    val sppModelPath = folder+"/assets"
    val savedModel = new File(folder, "saved_model.pb")
    val sppModel = new File(sppModelPath, "spiece.model")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )
    require(sppModel.exists(), s"SentencePiece model not found in folder $sppModelPath")

    val wrapper = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"))
    val spp = SentencePieceWrapper.read(sppModel.toString)

    val t5model = new T5Transformer().setModelIfNotSet(spark, wrapper, spp)

    t5model
  }
}


object T5Transformer extends ReadablePretrainedT5TransformerModel with ReadT5TransformerTensorflowModel with ReadSentencePieceModel