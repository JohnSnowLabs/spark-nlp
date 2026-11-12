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

package com.johnsnowlabs.nlp.annotators.ld.dl

import com.johnsnowlabs.ml.ai.{FastTextFeaturizer, FastTextLD}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID
import scala.collection.immutable.ListMap
import scala.io.Source

/** Language Identification and Detection by using CNN and RNN architectures in TensorFlow, or
  * supervised fastText models such as GlotLID on ONNX.
  *
  * `LanguageDetectorDL` is an annotator that detects the language of documents or sentences
  * depending on the inputCols. The CNN models are trained on large datasets such as Wikipedia and
  * Tatoeba. Depending on the language (how similar the characters are), they work best with text
  * longer than 140 characters. Their output is a language code in
  * [[https://en.wikipedia.org/wiki/List_of_Wikipedias Wiki Code style]].
  *
  * The fastText models run on ONNX and turn text into features exactly as fastText does. GlotLID
  * v4 (`"ld_glotlid_v4"`, or `"ld_glotlid_v4_int8"` at a quarter of the size) labels 2,060
  * languages and scripts as an ISO 639-3 code plus an ISO 15924 script, such as `eng_Latn` or
  * `cmn_Hani`, and has `zxx_Code`, `zxx_Zmth` and `zxx_Zsym` for code, mathematics and symbols.
  * For these models the metadata also holds `confidence`, the score the label was judged on, and
  * three parameters shape the result: `scriptPooling`, `minLetters` and `topK`. Each sentence
  * goes to the model whole, so split long documents into sentences first. To import other
  * fastText models, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/onnx/HuggingFace_ONNX_in_Spark_NLP_LanguageDetectorDL.ipynb import notebook]].
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * Val languageDetector = LanguageDetectorDL.pretrained()
  *   .setInputCols("sentence")
  *   .setOutputCol("language")
  * }}}
  * The default model is `"ld_wiki_tatoeba_cnn_21"`, default language is `"xx"` (meaning
  * multi-lingual), if no values are provided. For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Language+Detection Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/language-detection/Language_Detection_and_Indentification.ipynb Examples]],
  * the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/language-detection/Language_Detection_with_GlotLID.ipynb GlotLID example]]
  * And the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDLTestSpec.scala LanguageDetectorDLTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val languageDetector = LanguageDetectorDL.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("language")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     languageDetector
  *   ))
  *
  * val data = Seq(
  *   "Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages.",
  *   "Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.",
  *   "Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("language.result").show(false)
  * +------+
  * |result|
  * +------+
  * |[en]  |
  * |[fr]  |
  * |[de]  |
  * +------+
  * }}}
  *
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
class LanguageDetectorDL(override val uid: String)
    extends AnnotatorModel[LanguageDetectorDL]
    with HasSimpleAnnotate[LanguageDetectorDL]
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("LANGUAGE_DETECTOR_DL"))

  /** Alphabet used to feed the TensorFlow model for prediction
    *
    * @group param
    */
  val alphabet: MapFeature[String, Int] = new MapFeature(this, "alphabet").setProtected()

  /** Labels the model predicts, with their output index
    *
    * @group param
    */
  val language: MapFeature[String, Int] = new MapFeature(this, "language").setProtected()

  /** Dictionary of a fastText (ONNX) model: word to input-matrix row
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** Featurizer settings of a fastText (ONNX) model: `minn`, `maxn`, `bucket` and `nwords`
    *
    * @group param
    */
  val fastTextConfig: MapFeature[String, Int] =
    new MapFeature(this, "fastTextConfig").setProtected()

  /** The minimum threshold for the final result, otherwise it will be either `"unk"` or the value
    * set in `thresholdLabel` (Default: `0.1f`). Value is between 0.0 to 1.0. Try to set this
    * lower if your text is hard to predict
    *
    * @group param
    */
  val threshold = new FloatParam(
    this,
    "threshold",
    "The minimum threshold for the final result otherwise it will be either Unknown or the value set in thresholdLabel.")

  /** Value for the classification, if confidence is less than `threshold` (Default: `"unk"`).
    *
    * @group param
    */
  val thresholdLabel = new Param[String](
    this,
    "thresholdLabel",
    "In case the score is less than threshold, what should be the label. Default is Unknown.")

  /** Output average of sentences instead of one output per sentence (Default: `true`).
    *
    * @group param
    */
  val coalesceSentences = new BooleanParam(
    this,
    "coalesceSentences",
    "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.")

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Languages the model was trained with.
    *
    * @group param
    */
  val languages =
    new StringArrayParam(this, "languages", "keep an internal copy of languages for Python")

  /** Number of language scores kept in the metadata, the highest ones; 0 keeps all of them
    * (Default: `0`). ONNX (fastText) models only.
    *
    * @group param
    */
  val topK = new IntParam(
    this,
    "topK",
    "Number of language scores kept in the metadata, the highest ones; 0 keeps all of them")

  /** Whether the score of a non-Latin top label is the sum of all labels in its script, so that
    * close varieties such as Mandarin (`cmn_Hani`) and Cantonese (`yue_Hani`) do not split it
    * (Default: `false`). The label is still the top one, so the pooled score says how sure the
    * script is rather than the language. Labels must end with their ISO 15924 script, as in
    * `cmn_Hani`. ONNX (fastText) models only.
    *
    * @group param
    */
  val scriptPooling = new BooleanParam(
    this,
    "scriptPooling",
    "Whether the score of a non-Latin top label is the sum of all labels in its script")

  /** When the top label is in Latin script (`_Latn`), text with fewer letters than this gets
    * `thresholdLabel`; 0 turns it off (Default: `0`). Very short text such as "No oral lesions."
    * is often given a close language with a high score. ONNX (fastText) models only.
    *
    * @group param
    */
  val minLetters = new IntParam(
    this,
    "minLetters",
    "When the top label is in Latin script, text with fewer letters than this gets thresholdLabel; 0 turns it off")

  /** @group setParam */
  def setTopK(value: Int): this.type = {
    require(value >= 0, "topK must be 0 or more")
    set(topK, value)
  }

  /** @group getParam */
  def getTopK: Int = $(topK)

  /** @group setParam */
  def setScriptPooling(value: Boolean): this.type = set(scriptPooling, value)

  /** @group getParam */
  def getScriptPooling: Boolean = $(scriptPooling)

  /** @group setParam */
  def setMinLetters(value: Int): this.type = {
    require(value >= 0, "minLetters must be 0 or more")
    set(minLetters, value)
  }

  /** @group getParam */
  def getMinLetters: Int = $(minLetters)

  /** @group setParam */
  def setLanguage(value: Map[String, Int]): this.type = {
    set(this.language, value)
    this
  }

  /** @group setParam */
  def setAlphabet(value: Map[String, Int]): this.type = {
    set(alphabet, value)
    this
  }

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** @group setParam */
  def setFastTextConfig(value: Map[String, Int]): this.type = set(fastTextConfig, value)

  /** @group setParam */
  def setThreshold(threshold: Float): this.type = set(this.threshold, threshold)

  /** @group setParam */
  def setThresholdLabel(label: String): this.type = set(this.thresholdLabel, label)

  /** @group setParam */
  def setCoalesceSentences(value: Boolean): this.type = set(coalesceSentences, value)

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): LanguageDetectorDL.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getLanguage: Array[String] = {
    val langs = $$(language).toSeq.sortBy(_._2).map(_._1).toArray
    set(languages, langs)
    langs
  }

  /** @group getParam */
  def getThreshold: Float = $(this.threshold)

  /** @group getParam */
  def getThresholdLabel: String = $(this.thresholdLabel)

  /** @group getParam */
  def getCoalesceSentences: Boolean = $(coalesceSentences)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    inputCols -> Array("document"),
    outputCol -> "language",
    threshold -> 0.1f,
    thresholdLabel -> "unk",
    coalesceSentences -> true,
    topK -> 0,
    scriptPooling -> false,
    minLetters -> 0)

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
            configProtoBytes = getConfigProtoBytes,
            ListMap($$(language).toSeq.sortBy(_._2): _*),
            ListMap($$(alphabet).toSeq.sortBy(_._2): _*))))
    }

    this
  }

  private var _onnxModel: Option[Broadcast[FastTextLD]] = None

  /** @group getParam */
  def getOnnxModelIfNotSet: FastTextLD = _onnxModel.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, onnxWrapper: OnnxWrapper): this.type = {
    if (_onnxModel.isEmpty) {
      val config = $$(fastTextConfig)
      val featurizer = new FastTextFeaturizer(
        $$(vocabulary),
        nwords = config("nwords"),
        minn = config("minn"),
        maxn = config("maxn"),
        bucket = config("bucket"))
      val labels = $$(language).toSeq.sortBy(_._2).map(_._1).toArray
      _onnxModel = Some(
        spark.sparkContext.broadcast(new FastTextLD(onnxWrapper, featurizer, labels)))
    }
    this
  }

  /** Takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(annotations)
    val nonEmptySentences = sentences.filter(_.content.nonEmpty)
    if (nonEmptySentences.isEmpty) Seq.empty[Annotation]
    else if (getEngine == ONNX.name)
      getOnnxModelIfNotSet.predict(
        nonEmptySentences,
        $(threshold),
        $(thresholdLabel),
        $(coalesceSentences),
        $(scriptPooling),
        $(minLetters),
        $(topK))
    else
      getModelIfNotSet.predict(
        nonEmptySentences,
        $(threshold),
        $(thresholdLabel),
        $(coalesceSentences))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.LANGUAGE

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getOnnxModelIfNotSet.onnxWrapper,
          "_ld",
          LanguageDetectorDL.onnxFile)
      case _ =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflow,
          "_ld",
          LanguageDetectorDL.tfFile,
          configProtoBytes = getConfigProtoBytes)
    }
  }

}

trait ReadablePretrainedLanguageDetectorDLModel
    extends ParamsAndFeaturesReadable[LanguageDetectorDL]
    with HasPretrained[LanguageDetectorDL] {
  override val defaultModelName: Some[String] = Some("ld_wiki_tatoeba_cnn_21")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): LanguageDetectorDL = super.pretrained()

  override def pretrained(name: String): LanguageDetectorDL = super.pretrained(name)

  override def pretrained(name: String, lang: String): LanguageDetectorDL =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): LanguageDetectorDL =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadLanguageDetectorDLTensorflowModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[LanguageDetectorDL] =>

  override val tfFile: String = "ld_tensorflow"
  override val onnxFile: String = "ld_onnx"

  def readModel(instance: LanguageDetectorDL, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_ld_onnx", zipped = true, useBundle = false)
        instance.setModelIfNotSet(spark, onnxWrapper)
      case _ =>
        val tf = readTensorflowModel(path, spark, "_ld_tf")
        instance.setModelIfNotSet(spark, tf)
    }
    // This allows for Python to access getLanguages function, in output order
    instance.set(
      instance.languages,
      instance.language.get.get.toSeq.sortBy(_._2).map(_._1).toArray)
  }

  addReader(readModel)

  /** Loads a TensorFlow CNN export (`alphabet.txt`, `language.txt`) or a supervised fastText
    * model exported to ONNX: `model.onnx` with `assets/language.txt` (labels in output order),
    * `assets/vocab.json` (UTF-8, words in id order, `null` for a word that is not valid UTF-8)
    * and `assets/config.json` (`minn`, `maxn`, `bucket`, `nwords`).
    */
  def loadSavedModel(modelPath: String, spark: SparkSession): LanguageDetectorDL = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val labels = loadTextAsset(localModelPath, "language.txt")

    /*Universal parameters for all engines*/
    val annotatorModel = new LanguageDetectorDL()
      .setLanguage(labels.zipWithIndex.toMap)
    // This allows for Python to access getLanguages function
    annotatorModel.set(annotatorModel.languages, labels)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val alphabets = loadTextAsset(localModelPath, "alphabet.txt").zipWithIndex.toMap
        annotatorModel.setAlphabet(alphabets)

        val (wrapper, _) =
          TensorflowWrapper.read(
            localModelPath,
            zipped = false,
            useBundle = true,
            tags = Array("serve"))

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setModelIfNotSet(spark, wrapper)

      case ONNX.name =>
        implicit val formats: DefaultFormats.type = DefaultFormats
        val config = parse(loadJsonStringAsset(localModelPath, "config.json"))
          .extract[Map[String, Int]]
        val vocabFile = new File(s"$localModelPath/assets", "vocab.json")
        require(vocabFile.exists, s"File vocab.json not found in folder $localModelPath/assets")
        val vocabSource = Source.fromFile(vocabFile, "UTF-8")
        val words =
          try parse(vocabSource.mkString).children
          finally vocabSource.close()
        val configKeys = Set("minn", "maxn", "bucket", "nwords")
        configKeys.foreach(k => require(config.contains(k), s"assets/config.json has no $k"))
        require(
          config("nwords") == words.size,
          s"assets/vocab.json has ${words.size} words, config.json says ${config("nwords")}")
        val vocabulary = words.zipWithIndex.collect { case (JString(w), i) => w -> i }.toMap
        require(
          vocabulary.contains(FastTextFeaturizer.Eos),
          s"assets/vocab.json has no ${FastTextFeaturizer.Eos}")

        annotatorModel
          .setVocabulary(vocabulary)
          .setFastTextConfig(config.filter { case (k, _) => configKeys(k) })

        // Spark registers files by name, so two model.onnx exports would clash. The copy stays
        // for the executors until the JVM exits (deleteOnExit runs in reverse order)
        val uniqueName = UUID.randomUUID().toString.takeRight(12) + "_ld"
        val copyDir = Files.createTempDirectory(uniqueName)
        val copy = copyDir.resolve(s"$uniqueName.onnx")
        copyDir.toFile.deleteOnExit()
        copy.toFile.deleteOnExit()
        Files.copy(Paths.get(localModelPath, ONNX.modelName), copy)
        val onnxWrapper = OnnxWrapper.read(
          spark,
          copyDir.toString,
          zipped = false,
          useBundle = true,
          modelName = uniqueName)
        annotatorModel.setModelIfNotSet(spark, onnxWrapper)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[LanguageDetectorDL]]. Please refer to that class for the
  * documentation.
  */
object LanguageDetectorDL
    extends ReadablePretrainedLanguageDetectorDLModel
    with ReadLanguageDetectorDLTensorflowModel
