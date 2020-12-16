package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.tensorflow.sentencepiece.{ReadSentencePieceModel, SentencePieceWrapper, WriteSentencePieceModel}
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowT5, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, ParamsAndFeaturesWritable}

import scala.collection.mutable.Map
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.io.File

class T5Transformer(override val uid: String)
  extends AnnotatorModel[T5Transformer]
    with HasStorageRef
    with ParamsAndFeaturesWritable
    with WriteTensorflowModel
    with WriteSentencePieceModel {

  def this() = this(Identifiable.randomUID("SentenceDetectorDLModel"))

  /** Output annotator type : TOKEN
    *
    * @group anno
    * */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    * */
  override val outputAnnotatorType: String = DOCUMENT

  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  var task = new Param[String](this, "task", "Set transformer task, e.g. 'summarize'.")

  private var _tfModel: Option[Broadcast[TensorflowT5]] = None

  def setConfigProtoBytes(bytes: Array[Int]): T5Transformer  .this.type = set(this.configProtoBytes, bytes)

  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  def setTask(taskPrefix: String): T5Transformer.this.type = {
    if (get(task).isEmpty)
      set(task, taskPrefix)
    this
  }

  setDefault(
    task -> ""
  )

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

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    var sentences: Array[String] = Array()
    annotations.filter(anno => this.inputAnnotatorTypes.contains(anno.annotatorType)).foreach(anno => {
      val sentenceNo = anno.metadata("sentence").toInt
      while (sentenceNo >= sentences.length) {
        sentences = sentences ++ Array(get(task).getOrElse(""))
      }
      sentences(sentenceNo) = sentences.last.concat(" ").concat(anno.result)
    })

    this.getModelIfNotSet.process(sentences).zipWithIndex.map(x => {
      new Annotation(
        annotatorType = this.outputAnnotatorType,
        begin = 0,
        end = x._1.length,
        result = x._1,
        metadata = Map("sentence" -> x._2.toString))
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_t5", T5Transformer.tfFile, configProtoBytes = getConfigProtoBytes)
    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, "_t5",  T5Transformer.sppFile)

  }
}

trait ReadablePretrainedT5TransformerModel extends ParamsAndFeaturesReadable[T5Transformer] with HasPretrained[T5Transformer] {
  override val defaultModelName: Some[String] = Some("albert_base_uncased")
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
    val tf = readTensorflowModel(path, spark, "_t5_tf", initAllTables = false)
    val spp = readSentencePieceModel(path, spark, "_t5_spp" )
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