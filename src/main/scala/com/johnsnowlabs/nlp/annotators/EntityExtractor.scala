package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, DocumentAssembler}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.param.{BooleanParam, Param}

class EntityExtractor(override val uid: String) extends AnnotatorApproach[EntityExtractorModel] {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  override val requiredAnnotatorTypes = Array(TOKEN)

  override val annotatorType: AnnotatorType = ENTITY

  override val description: String = "Extracts entities from target dataset given in a text file"

  val entitiesPath = new Param[String](this, "entitiesPath", "Path to entities (phrases) to extract")
  val entitiesFormat = new Param[String](this, "entitiesFormat", "TXT or TXTDS for reading as dataset")

  setDefault(
    entitiesFormat -> "TXT",
    inputCols -> Array(TOKEN)
  )

  def setEntitiesPath(value: String): this.type = set(entitiesPath, value)

  def getEntitiesPath: String = $(entitiesPath)

  def setEntitiesFormat(value: String): this.type = set(entitiesFormat, value)

  def getEntitiesFormat: String = $(entitiesFormat)

  /**
    * Loads entities from a provided source.
    */
  private def loadEntities(): Array[Array[String]] = {
    val phrases: Array[String] = ResourceHelper.parseLinesText($(entitiesPath), $(entitiesFormat).toUpperCase())
    val tokenizer = new Tokenizer()
    val entities: Array[Array[String]] = phrases.map {
      line =>
        val annotation = Seq(Annotation(line))
        val tokens = tokenizer.annotate(annotation)
        tokens.map(_.result).toArray
    }
    entities
  }

  private def loadEntities(pipelineModel: PipelineModel): Array[Array[String]] = {
    val phrases: Seq[String] = ResourceHelper.parseLinesText($(entitiesPath), $(entitiesFormat).toUpperCase())
    import ResourceHelper.spark.implicits._
    val textColumn = pipelineModel.stages.find {
      case _: DocumentAssembler => true
      case _ => false
    }.map(_.asInstanceOf[DocumentAssembler].getInputCol)
      .getOrElse(throw new Exception("Could not retrieve DocumentAssembler from RecursivePipeline"))
    val data = phrases.toDS.withColumnRenamed("value", textColumn)
    pipelineModel.transform(data).select($(inputCols).head).as[Array[Annotation]].map(_.map(_.result)).collect
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): EntityExtractorModel = {
    if (recursivePipeline.isDefined) {
      new EntityExtractorModel()
        .setEntities(loadEntities(recursivePipeline.get))
    } else {
      new EntityExtractorModel()
        .setEntities(loadEntities())
    }
  }

}

object EntityExtractor extends DefaultParamsReadable[EntityExtractor]