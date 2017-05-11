package com.jsl.nlp

import com.jsl.nlp.annotators._
import com.jsl.nlp.util.RegexRule
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.{FlatSpec, Suite}

/**
  * Created by saif on 02/05/17.
  * Generates different Annotator pipeline paths
  * Place to add different annotator constructions
  */
object AnnotatorBuilder extends FlatSpec with SparkBasedTest { this: Suite =>

  def withTokenizer(dataset: Dataset[Row]): Dataset[Row] = {
    val regexTokenizer = new RegexTokenizer()
      .setDocumentCol("document")
      .setPattern("[a-zA-Z]+|[0-9]+|\\p{Punct}")
    regexTokenizer.transform(dataset)
  }

  def withFullStemmer(dataset: Dataset[Row]): Dataset[Row] = {
    val stemmer = new Stemmer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("token"))
    stemmer.transform(withTokenizer(dataset))
  }

  def withFullNormalizer(dataset: Dataset[Row]): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("stem"))
    normalizer.transform(withFullStemmer(dataset))
  }

  def withFullLemmatizer(dataset: Dataset[Row]): Dataset[Row] = {
    val lemmatizer = new Lemmatizer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("ntoken"))
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
      .setInputAnnotationCols(Array("ntoken"))
      .setMaxLen(4)
      .setEntities(entities)
    entityExtractor.transform(withFullLemmatizer(dataset))
  }

  def withRegexMatcher(dataset: Dataset[Row], rules: Seq[RegexRule]): Dataset[Row] = {
    val regexMatcher = new RegexMatcher()
      .setDocumentCol("document")
      .setPatterns(rules)
    regexMatcher.transform(dataset)
  }

}
