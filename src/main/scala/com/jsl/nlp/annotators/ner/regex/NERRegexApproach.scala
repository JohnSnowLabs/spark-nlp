package com.jsl.nlp.annotators.ner.regex

import com.jsl.nlp.AnnotatorApproach
import com.jsl.nlp.util.io.ResourceHelper
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.param.{Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class NERRegexApproach(override val uid: String) extends AnnotatorApproach[NERRegexModel] {

  import com.jsl.nlp.AnnotatorType._
  override val description: String = "Dictionary NER Tagger"

  val corpusPath = new Param[String](this, "corpusPath", "path to corpus files")
  val corpus = new StringArrayParam(this, "corpus", "corpus content")

  def this() = this(Identifiable.randomUID("NER"))

  override val annotatorType: AnnotatorType = NAMED_ENTITY

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  def setCorpusPath(value: String): this.type = set(corpusPath, value)

  def setCorpus(value: Array[String]): this.type = set(corpus, value)

  override def train(dataset: Dataset[_]): NERRegexModel = {
    if (get(corpusPath).isDefined) {
      NERRegexApproach.train($(corpusPath))
    } else {
      NERRegexApproach.train($(corpus))
    }
  }
}

object NERRegexApproach extends DefaultParamsReadable[NERRegexApproach] {

  private val config: Config = ConfigFactory.load

  /**
    *
    * @param entitiesPath The path to load the dictionary from
    * @param fileFormat The format of the file specified at the path
    * @return The dictionary as a Map
    */
  private def retrieveEntityDict(entitiesPath: String = "__default",
                         fileFormat: String = config.getString("nlp.entityRecognition.format")
                        ): Map[String, String] = {
    val filePath = if (entitiesPath == "__default") config.getString("nlp.entityRecognition.file") else entitiesPath
    ResourceHelper.parseKeyValueText(filePath, fileFormat, ":")
  }

  private def retrieveEntityDicts(files: Array[String],
                          fileFormat: String = config.getString("nlp.entityRecognition.format")
                         ): Map[String, String] = {

    files.map( f => ResourceHelper.parseKeyValueText(f, fileFormat, ":") ).foldRight(Map[String, String]())( (m1, m2) => m1 ++ m2)
  }

  def train(path: String = "__default"): NERRegexModel = {
    val entityDictionary = retrieveEntityDict(path)
    new NERRegexModel().setModel(entityDictionary)
  }

  def train(files: Array[String]): NERRegexModel = {
    val entityDictionary = retrieveEntityDicts(files)
    new NERRegexModel().setModel(entityDictionary)
  }
  def train(dict: Map[String, String]): NERRegexModel = {
    new NERRegexModel().setModel(dict)
  }
}
