package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import com.johnsnowlabs.util.JsonParser
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

import scala.io.Source

class EntityRuler(override val uid: String) extends AnnotatorApproach[EntityRulerModel] with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  override val description: String = "Entity Ruler matches entities based on text patterns"

  val patterns = new ExternalResourceParam(this, "patterns",
    "Resource in JSON or CSV format to map entities to patterns")

  def setPatterns(path: String, readAs: ReadAs.Format, options: Map[String, String] = Map("format" -> "JSON")): this.type =
    set(patterns, ExternalResource(path, readAs, options))

  setDefault(storagePath -> ExternalResource("", ReadAs.TEXT, Map()),
    patterns -> null
  )

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): EntityRulerModel = {

    new EntityRulerModel()
      .setStorageRef($(storageRef))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)
  override val outputAnnotatorType: AnnotatorType = CHUNK
  override protected val databases: Array[Name] = EntityRulerModel.databases

  override protected def index(fitDataset: Dataset[_], storageSourcePath: Option[String], readAs: Option[ReadAs.Value],
                               writers: Map[Name, StorageWriter[_]], readOptions: Option[Map[String, String]]): Unit = {

    require($(patterns) != null, "patterns parameter required")

    val patternsReaderWriter = writers(Database.ENTITY_PATTERNS).asInstanceOf[PatternsReadWriter]

    entityPatterns.foreach{ case (pattern, entity) =>
      patternsReaderWriter.lookup(pattern).getOrElse(patternsReaderWriter.add(pattern, entity))
    }
  }

  private lazy val entityPatterns: Map[String, String] = {

    import io.circe.generic.auto._

    val stream =  ResourceHelper.getResourceStream($(patterns).path)
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]
    val entityPatterns: Array[EntityPattern] = jsonParser.readJsonArray(jsonContent)

    entityPatterns.flatMap{ entityPattern =>
      entityPattern.pattern.split("\\s").flatMap(pattern => Map(pattern -> entityPattern.label))
    }.toMap
  }

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new PatternsReadWriter(connection)
  }
}