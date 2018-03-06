package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File
import java.nio.file.Files
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.{DatasetEncoder, DatasetEncoderParams, TensorflowNer}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.TokenizedWithSentence
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.tensorflow.{Graph, Session}


class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with HasWordEmbeddings
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN)
  override val annotatorType = NAMED_ENTITY

  val settings: StructFeature[DatasetEncoderParams] = new StructFeature[DatasetEncoderParams](this, "encoderParams")
  def setParams(params: DatasetEncoderParams): NerDLModel = set(settings, params)

  @transient
  var session: Option[Session] = None
  @transient
  var graph: Option[Graph] = None

  def setSession(session: Session, graph: Graph): NerDLModel = {
    this.session = Some(session)
    this.graph = Some(graph)

    this
  }

  @transient
  lazy val model = {
    require(this.settings.isSet, "set settings before model usage")
    require(this.session.isDefined, "set session before model usage")

    val settings = get(this.settings).get
    val encoder = new DatasetEncoder(
      embeddings.get.getEmbeddings,
      settings
    )

    new TensorflowNer(
      session.get,
      encoder,
      10,
      Verbose.Silent
    )
  }


  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    // Parse
    val tokenized = TokenizedWithSentence.unpack(annotations).toArray

    // Predict
    val labels = model.predict(tokenized)

    // Combine labels with sentences tokens
    (0 until tokenized.length).flatMap{i =>
      val sentence = tokenized(i)
      (0 until sentence.tokens.length).map{j =>
        val token = sentence.indexedTokens(j)
        val label = labels(i)(j)
        new Annotation(NAMED_ENTITY, token.begin, token.end, label, Map.empty)
      }
    }
  }


  override def onWrite(path: String, spark: SparkSession): Unit = {
    val session = this.session.get
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    val modelName = "model_temp_e08d7ffea7a143eeabde4444e8d81456"
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_nerdl")
      .toAbsolutePath.toString

    // 1. Save variables
    session.runner.addTarget("save/SaveV2")
    FileUtils.moveDirectory(new File(modelName), new File(tmpFolder + "/variables"))

    // 2. Save Graph
    val graphDef = graph.get.toGraphDef
    FileUtils.writeByteArrayToFile(new File("saved_model.pb"), graphDef)

    // 3. Find save model and upload to path
    fs.copyFromLocalFile(new Path(tmpFolder), new Path(path))
  }
}

object NerDLModel extends ParamsAndFeaturesReadable[NerDLModel] {

  override def onRead(instance: NerDLModel, path: String, spark: SparkSession): Unit = {
    val bundle = NerDLModelPythonReader.readBundle(path, spark)

    instance
      .setSession(bundle.session, bundle.graph)
  }
}
