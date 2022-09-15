package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow.{MergeTokenStrategy, ReadTensorflowModel, TensorflowBertClassification, TensorflowTapas, TensorflowWrapper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, HasPretrained, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.io.File

class TapasForQuestionAnswering(override val uid: String) extends BertForQuestionAnswering(uid) {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("TapasForQuestionAnswering"))


  /** Input Annotator Types: DOCUMENT, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT)

  private var _model: Option[Broadcast[TensorflowTapas]] = None

  /** @group setParam */
  override def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: TensorflowWrapper): TapasForQuestionAnswering = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowTapas(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            tags = Map.empty[String, Int],
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  override def getModelIfNotSet: TensorflowTapas = _model.get.value


  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    batchedAnnotations.map(annotations => {

      val documents = annotations
        .filter(_.annotatorType == AnnotatorType.DOCUMENT)
        .toSeq

      if (documents.nonEmpty) {
        getModelIfNotSet.predictSpan(
          documents,
          $(maxSentenceLength),
          $(caseSensitive),
          MergeTokenStrategy.vocab)
      } else {
        Seq.empty[Annotation]
      }
    })
  }

}

trait ReadablePretrainedTapasForQAModel
  extends ParamsAndFeaturesReadable[TapasForQuestionAnswering]
    with HasPretrained[TapasForQuestionAnswering] {
  override val defaultModelName: Some[String] = Some("tapas")

  /** Java compliant-overrides */
  override def pretrained(): TapasForQuestionAnswering = super.pretrained()

  override def pretrained(name: String): TapasForQuestionAnswering = super.pretrained(name)

  override def pretrained(name: String, lang: String): TapasForQuestionAnswering =
    super.pretrained(name, lang)

  override def pretrained(
                           name: String,
                           lang: String,
                           remoteLoc: String): TapasForQuestionAnswering = super.pretrained(name, lang, remoteLoc)
}

trait ReadTapasForQATensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[TapasForQuestionAnswering] =>

  override val tfFile: String = "bert_classification_tensorflow"

  def readTensorflow(
                      instance: TapasForQuestionAnswering,
                      path: String,
                      spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_bert_classification_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): TapasForQuestionAnswering = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath")

    val vocabPath = new File(tfModelPath + "/assets", "vocab.txt")
    require(
      vocabPath.exists(),
      s"Vocabulary file vocab.txt not found in folder $tfModelPath/assets/")

    val vocabResource =
      new ExternalResource(vocabPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val (wrapper, signatures) =
      TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new TapasForQuestionAnswering()
      .setVocabulary(words)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}

/** This is the companion object of [[BertForQuestionAnswering]]. Please refer to that class for
  * the documentation.
  */
object TapasForQuestionAnswering
  extends ReadablePretrainedTapasForQAModel
    with ReadTapasForQATensorflowModel
