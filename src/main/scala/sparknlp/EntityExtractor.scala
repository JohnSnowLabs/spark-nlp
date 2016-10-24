package sparknlp

import java.io.{FileInputStream, InputStream}

import org.apache.spark.ml.param.Param

/**
  * Created by alext on 10/23/16.
  */
class EntityExtractor(fromSentences: Boolean = false) extends Annotator {

  override val aType: String = "entity"

  override def annotate(
    document: Document, annos: Seq[Annotation]
  ): Seq[Annotation] =
    if (fromSentences) {
      annos.filter {
        case token: Annotation => token.aType == "sentence"
      }.flatMap {
        sentence =>
          val ntokens = annos.filter {
            case token: Annotation =>
              token.aType == "ntoken" &&
                token.begin >= sentence.begin &&
                token.end <= sentence.end
          }
          EntityExtractor.phraseMatch(ntokens, $(maxLen), $(entities))
      }
    } else {
      val ntokens = annos.filter {
        case token: Annotation => token.aType == "ntoken"
      }
      EntityExtractor.phraseMatch(ntokens, $(maxLen), $(entities))
    }

  override val requiredAnnotationTypes: Seq[String] =
    if (fromSentences) {
      Seq("sentence")
    } else {
      Seq()
    }

  val entities: Param[Set[Seq[String]]] = new Param(this, "entities", "set of entities (phrases)")

  def setEntities(value: Set[Seq[String]]): this.type = set(entities, value)

  def getEntities: Set[Seq[String]] = $(entities)

  val maxLen: Param[Int] = new Param(this, "maxLen", "maximum phrase length")

  def setMaxLen(value: Int): this.type = set(maxLen, value)

  def getMaxLen: Int = $(maxLen)
}

object EntityExtractor {
  def loadEntities(inputStream: InputStream, tokenPattern: String): Set[Seq[String]] = {
    val src = scala.io.Source.fromInputStream(inputStream)
    val tokenizer = new RegexTokenizer().setPattern(tokenPattern)
    val stemmer = new Stemmer()
    val normalizer = new Normalizer()
    val phrases: Set[Seq[String]] = src.getLines.map {
      line =>
        val doc = Document("", line)
        val tokens = tokenizer.annotate(doc, Seq())
        val stems = stemmer.annotate(doc, tokens)
        val ntokens = normalizer.annotate(doc, stems)
        ntokens.map(_.metadata("ntoken"))
    }.toSet
    src.close()
    phrases
  }

  def loadEntities(path: String, tokenPattern: String): Set[Seq[String]] = {
    loadEntities(new FileInputStream(path), tokenPattern)
  }

  def phraseMatch(ntokens: Seq[Annotation], maxLen: Int, entities: Set[Seq[String]]): Seq[Annotation] = {
    ntokens.padTo(ntokens.length + maxLen - (ntokens.length % maxLen), null).sliding(maxLen).flatMap {
      window =>
        window.filter(_ != null).inits.filter {
          phraseCandidate =>
            entities.contains(phraseCandidate.map(_.metadata("ntoken")))
        }.map {
          phrase =>
            Annotation(
              "entity",
              phrase.head.begin,
              phrase.last.end,
              Map("entity" -> phrase.map(_.metadata("ntoken")).mkString(" "))
            )
        }
    }.toSeq
  }
}