package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp.util.ConfigHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Tokenized}
import com.typesafe.config.Config
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import com.johnsnowlabs.nlp.AnnotatorType._

import scala.collection.mutable.ArrayBuffer


/**
  * Extracts entities out of provided phrases
  * @param uid internally required UID to make it writable
  * @@ entitiesPath: Path to file with phrases to search
  * @@ insideSentences: Should Extractor search only within sentence borders?
  */
class EntityExtractor(override val uid: String) extends AnnotatorModel[EntityExtractor] {

  private val config: Config = ConfigHelper.retrieve

  val entitiesPath = new Param[String](this, "entitiesPath", "Path to entities (phrases) to extract")
  val entitiesFormat = new Param[String](this, "entitiesFormat", "TXT or TXTDS for reading as dataset")

  if (config.getString("nlp.entityExtractor.file").nonEmpty)
    setDefault(entitiesPath, config.getString("nlp.entityExtractor.file"))

  setDefault(entitiesFormat, config.getString("nlp.entityExtractor.format"))

  val insideSentences = new BooleanParam(this, "insideSentences",
    "Should extractor search only within sentences borders?")

  override val annotatorType: AnnotatorType = ENTITY

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN)

  setDefault(
    inputCols -> Array(DOCUMENT, TOKEN),
    insideSentences -> true
  )

  /** internal constructor for writabale annotator */
  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  def setEntitiesPath(value: String): this.type = set(entitiesPath, value)

  def getEntitiesPath: String = $(entitiesPath)

  def setEntitiesFormat(value: String): this.type = set(entitiesFormat, value)

  def getEntitiesFormat: String = $(entitiesFormat)

  def setInsideSentences(value: Boolean): this.type = set(insideSentences, value)

  def getEntities: Array[Array[String]] = {
    if (loadedPath != get(entitiesPath))
      loadEntities()
    loadedEntities
  }

  def getSearchTrie: SearchTrie = {
    if (loadedPath != get(entitiesPath))
      loadEntities()

    searchTrie
  }

  private var loadedEntities = Array.empty[Array[String]]
  private var loadedPath = get(entitiesPath)
  private var searchTrie = SearchTrie(Array.empty)

  /**
    * Loads entities from a provided source.
    */
  private def loadEntities(): Unit = {
    val src = EntityExtractor.retrieveEntityExtractorPhrases($(entitiesPath), $(entitiesFormat))

    val tokenizer = new RegexTokenizer().setPattern("\\w+")
    val normalizer = new Normalizer()
    val phrases: Array[Array[String]] = src.map {
      line =>
        val annotation = Seq(Annotation(line))
        val tokens = tokenizer.annotate(annotation)
        val nTokens = normalizer.annotate(tokens)
        nTokens.map(_.result).toArray
    }

    loadedEntities = phrases
    searchTrie = SearchTrie.apply(loadedEntities)
    loadedPath = get(entitiesPath)
  }

  /**
    * Searches entities and stores them in the annotation
    * @param text Tokenized text to search
    * @return Extracted Entities
    */
  private def search(text: Array[IndexedToken]): Seq[Annotation] = {
    val words = text.map(t => t.token)
    val result = ArrayBuffer[Annotation]()

    for ((begin, end) <- getSearchTrie.search(words)) {
      val normalizedText = (begin to end).map(i => words(i)).mkString(" ")

      val annotation = Annotation(
        ENTITY,
        text(begin).begin,
        text(end).end,
        normalizedText,
        Map()
      )

      result.append(annotation)
    }

    result
  }

  /** Defines annotator phrase matching depending on whether we are using SBD or not */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokens = annotations.flatMap {
      case a@Annotation(AnnotatorType.TOKEN, _, _, _, _) =>
        Seq(a)
      case a => Some(a)
    }

    val sentences = Tokenized.unpack(tokens)
    if ($(insideSentences)) {
      sentences.flatMap(sentence => search(sentence.indexedTokens))
    } else {
      val allTokens = sentences.flatMap(s => s.indexedTokens).toArray
      search(allTokens)
    }
  }

}

object EntityExtractor extends DefaultParamsReadable[EntityExtractor] {

  protected def retrieveEntityExtractorPhrases(
                                                entitiesPath: String,
                                                fileFormat: String
                                              ): Array[String] = {
    ResourceHelper.parseLinesText(entitiesPath, fileFormat.toUpperCase())
  }
}