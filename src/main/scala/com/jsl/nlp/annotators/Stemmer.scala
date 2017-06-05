package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import org.tartarus.snowball.SnowballStemmer

/**
  * Created by alext on 10/23/16.
  */
class Stemmer(algorithm: String = "english") extends Annotator {

  private val stemmer: (String => String) = Stemmer.getStemmer(algorithm)

  override val aType: String = Stemmer.aType

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case token: Annotation if token.aType == RegexTokenizer.aType =>
        val stem = stemmer(document.text.substring(token.begin, token.end))
        Annotation(aType, token.begin, token.end, Map(aType -> stem))
    }

  override val requiredAnnotationTypes = Array(RegexTokenizer.aType)
}

object Stemmer {
  val aType = "stem"

  private def getStemmer(algorithm: String = "english"): (String => String) = {
    val stemmerClass = if (algorithm.indexOf('.') == -1) {
      Class.forName(s"org.tartarus.snowball.ext.${algorithm.toLowerCase()}Stemmer")
    } else {
      Class.forName(algorithm)
    }
    (token: String) => {
      val stemmer = stemmerClass.newInstance().asInstanceOf[SnowballStemmer]
      stemmer.setCurrent(token)
      stemmer.stem()
      stemmer.getCurrent
    }
  }
}