package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class EntityRuler(override val uid: String) extends AnnotatorApproach[EntityRulerModel] with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  override val description: String = "Entity Ruler matches entities based on text patterns"

  setDefault(storagePath -> new ExternalResource("", ReadAs.TEXT, Map()))

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

    val tmpEntityPatterns = Map("John" -> "PER", "Snow" -> "PER", "John Snow" -> "PER")

    val patternsReaderWriter = writers(Database.ENTITY_PATTERNS).asInstanceOf[PatternsReadWriter]

    tmpEntityPatterns.foreach{ case (pattern, entity) =>
      patternsReaderWriter.lookup(pattern).getOrElse(patternsReaderWriter.add(pattern, entity))
    }
  }

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new PatternsReadWriter(connection)
  }
}
