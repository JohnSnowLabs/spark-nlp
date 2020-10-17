package com.johnsnowlabs.nlp.annotators.ld.dl

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/**
  * Language Identification by using Deep Neural Network in TensowrFlow and Keras
  * LanguageDetectorDL is an annotator that detects the language of documents or sentenccecs depending on the inputCols
  *
  * The models are trained on large datasets from Wikipedia
  * The output is a language code in Wiki Code style: https://en.wikipedia.org/wiki/List_of_Wikipedias
  *
  *
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class LanguageDetectorDL(override val uid: String) extends
  AnnotatorModel[LanguageDetectorDL]
  with WriteTensorflowModel {

  def this() = this(Identifiable.randomUID("LANGUAGE_DETECTOR_DL"))

  /** alphabet
    *
    * @group param
    **/
  val alphabet: MapFeature[String, Int] = new MapFeature(this, "alphabet")

  /** language
    *
    * @group param
    **/
  val language: MapFeature[String, Int] = new MapFeature(this, "language")

  /** threshold
    *
    * @group param
    **/
  val threshold = new FloatParam(this, "threshold", "The minimum threshold for the final result otheriwse it will be either Unknown or the value set in thresholdLabel.")

  /** thresholdLabel
    *
    * @group param
    **/
  val thresholdLabel = new Param[String](this, "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is Unknown.")

  /** coalesceSentences
    *
    * @group param
    **/
  val coalesceSentences = new BooleanParam(this, "coalesceSentences", "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.")

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** language used to map prediction to two-letter (ISO 639-1) language codes
    *
    * @group setParam
    * */
  def setLanguage(value: Map[String, Int]): this.type = {
    if (get(language).isEmpty)
      set(this.language, value)
    this
  }

  /** alphabet used to feed the TensorFlow model for prediction
    *
    * @group setParam
    * */
  def setAlphabet(value: Map[String, Int]): this.type = {
    if (get(language).isEmpty)
      set(alphabet, value)
    this
  }

  /** The minimum threshold for the final result otheriwse it will be either Unknown or the value set in thresholdLabel.
    *
    * @group setParam
    * */
  def setThreshold(threshold: Float): this.type = set(this.threshold, threshold)

  /** In case the score of prediction is less than threshold, what should be the label. Default is Unknown.
    *
    * @group setParam
    * */
  def setThresholdLabel(label: String):this.type = set(this.thresholdLabel, label)

  /** If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to false.
    *
    * @group setParam
    * */
  def setCoalesceSentences(value: Boolean): this.type = set(coalesceSentences, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group setParam
    * */
  def setConfigProtoBytes(bytes: Array[Int]): LanguageDetectorDL.this.type = set(this.configProtoBytes, bytes)

  /** threshold
    *
    * @group getParam
    **/
  def getThreshold: Float = $(this.threshold)

  /**
    *
    * @group thresholdLabel
    **/
  def getThresholdLabel: String = $(this.thresholdLabel)

  /**
    *
    * @group thresholdLabel
    **/
  def getCoalesceSentences: Boolean = $(coalesceSentences)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group getParam
    **/
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    inputCols-> Array("document"),
    outputCol-> "language",
    threshold -> 0.5f,
    thresholdLabel -> "Unknown",
    coalesceSentences -> true
  )

  private var _model: Option[Broadcast[TensorflowLD]] = None

  /** @group getParam */
  def getModelIfNotSet: TensorflowLD = _model.get.value
  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowLD(
            tensorflow,
            configProtoBytes = getConfigProtoBytes
          )
        )
      )
    }

    this
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(annotations)
    val nonEmptySentences = sentences.filter(_.content.nonEmpty)
    if (nonEmptySentences.nonEmpty) {
      getModelIfNotSet.calculateLanguageIdentification(
        nonEmptySentences,
        $$(alphabet),
        $$(language),
        $(threshold),
        $(thresholdLabel),
        $(coalesceSentences)
      )
    } else {
      Seq.empty[Annotation]
    }
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.LANGUAGE

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_ld", LanguageDetectorDL.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedLanguageDetectorDLModel extends ParamsAndFeaturesReadable[LanguageDetectorDL] with HasPretrained[LanguageDetectorDL] {
  override val defaultModelName: Some[String] = Some("ld_wiki_20")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): LanguageDetectorDL = super.pretrained()
  override def pretrained(name: String): LanguageDetectorDL = super.pretrained(name)
  override def pretrained(name: String, lang: String): LanguageDetectorDL = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): LanguageDetectorDL = super.pretrained(name, lang, remoteLoc)
}

trait ReadLanguageDetectorDLTensorflowModel extends ReadTensorflowModel {
  this:ParamsAndFeaturesReadable[LanguageDetectorDL] =>

  override val tfFile: String = "ld_tensorflow"

  def readTensorflow(instance: LanguageDetectorDL, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_ld_tf")
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): LanguageDetectorDL = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val alphabetPath = new File(folder+"/assets", "alphabet.txt")
    val languagePath = new File(folder+"/assets", "language.txt")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(alphabetPath.exists(), s"Alphabet file alphabet.txt not found in folder $folder")
    require(languagePath.exists(), s"Language file language.txt not found in folder $folder")

    val alphabetResource = new ExternalResource(alphabetPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val alphabets = ResourceHelper.parseLines(alphabetResource).zipWithIndex.toMap

    val languageResource = new ExternalResource(languagePath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val languages = ResourceHelper.parseLines(languageResource).zipWithIndex.toMap

    val wrapper = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"))

    new LanguageDetectorDL()
      .setAlphabet(alphabets)
      .setLanguage(languages)
      .setModelIfNotSet(spark, wrapper)
  }
}


object LanguageDetectorDL extends ReadablePretrainedLanguageDetectorDLModel with ReadLanguageDetectorDLTensorflowModel
