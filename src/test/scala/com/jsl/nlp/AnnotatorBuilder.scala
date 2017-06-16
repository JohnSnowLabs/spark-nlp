package com.jsl.nlp

import com.jsl.nlp.annotators._
import com.jsl.nlp.annotators.pos.POSTagger
import com.jsl.nlp.annotators.pos.perceptron.PerceptronApproach
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.annotators.sbd.pragmatic.PragmaticApproach
import com.jsl.nlp.util.ResourceHelper
import com.jsl.nlp.util.regex.RegexRule
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  * Generates different Annotator pipeline paths
  * Place to add different annotator constructions
  */
object AnnotatorBuilder extends FlatSpec { this: Suite =>

  def withTokenizer(dataset: Dataset[Row]): Dataset[Row] = {
    val regexTokenizer = new RegexTokenizer()
      .setDocumentCol("document")
      .setPattern("[a-zA-Z]+|[0-9]+|\\p{Punct}")
    regexTokenizer.transform(dataset)
  }

  def withFullStemmer(dataset: Dataset[Row]): Dataset[Row] = {
    val stemmer = new Stemmer()
      .setDocumentCol("document")
    stemmer.transform(withTokenizer(dataset))
  }

  def withFullNormalizer(dataset: Dataset[Row]): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setDocumentCol("document")
    normalizer.transform(withFullStemmer(dataset))
  }

  def withFullLemmatizer(dataset: Dataset[Row]): Dataset[Row] = {
    val lemmatizer = new Lemmatizer()
      .setDocumentCol("document")
    lemmatizer.transform(withFullNormalizer(dataset))
  }

  def withFullEntityExtractor(dataset: Dataset[Row]): Dataset[Row] = {
    val tokenPattern = "[a-zA-Z]+|[0-9]+|\\p{Punct}"
    val entities = EntityExtractor
      .loadEntities(
        getClass.getResourceAsStream("/test-phrases.txt"),
        tokenPattern)
    val entityExtractor = new EntityExtractor()
      .setDocumentCol("document")
      .setMaxLen(4)
      .setEntities(entities)
    entityExtractor.transform(withFullLemmatizer(dataset))
  }

  def withFullPragmaticSentenceDetector(dataset: Dataset[Row]): Dataset[Row] = {
    val pragmaticDetection = new PragmaticApproach
    val sentenceDetector = new SentenceDetector(pragmaticDetection)
      .setDocumentCol("document")
    sentenceDetector.transform(dataset)
  }

  def withFullPOSTagger(dataset: Dataset[Row]): Dataset[Row] = {
    val perceptronApproach = PerceptronApproach.train(
      ResourceHelper.parsePOSCorpusFromText(ContentProvider.wsjTrainingCorpus, '|')
    )
    val posTagger = new POSTagger(perceptronApproach)
      .setDocumentCol("document")
    posTagger.transform(withFullPragmaticSentenceDetector(dataset))
  }

  def withRegexMatcher(dataset: Dataset[Row], rules: Seq[(String, String)], strategy: String): Dataset[Row] = {
    val regexMatcher = new RegexMatcher()
      .setDocumentCol("document")
      .setPatterns(rules)
      .setStrategy(strategy)
    regexMatcher.transform(dataset)
  }

  def withDateMatcher(dataset: Dataset[Row]): Dataset[Row] = {
    val dateMatcher = new DateMatcher()
      .setDocumentCol("document")
    dateMatcher.transform(dataset)
  }

}
