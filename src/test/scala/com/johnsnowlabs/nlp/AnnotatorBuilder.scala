package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParser
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  * Generates different Annotator pipeline paths
  * Place to add different annotator constructions
  */
object AnnotatorBuilder extends FlatSpec { this: Suite =>

  def withDocumentAssembler(dataset: Dataset[Row]): Dataset[Row] = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
    documentAssembler.transform(dataset)
  }

  def withTokenizer(dataset: Dataset[Row]): Dataset[Row] = {
    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
    regexTokenizer.transform(withFullPragmaticSentenceDetector(dataset))
  }

  def withFullStemmer(dataset: Dataset[Row]): Dataset[Row] = {
    val stemmer = new Stemmer()
      .setInputCols(Array("token"))
      .setOutputCol("stem")
    stemmer.transform(withTokenizer(dataset))
  }

  def withFullNormalizer(dataset: Dataset[Row]): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
    normalizer.transform(withTokenizer(dataset))
  }

  def withFullLemmatizer(dataset: Dataset[Row]): Dataset[Row] = {
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
      .setLemmaDict("/lemma-corpus/AntBNC_lemmas_ver_001.txt")
    val tokenized = withTokenizer(dataset)
    lemmatizer.transform(tokenized)
  }

  def withFullEntityExtractor(dataset: Dataset[Row], insideSentences: Boolean = true): Dataset[Row] = {
    val entityExtractor = new EntityExtractor()
      .setInputCols("sentence", "normalized")
      .setInsideSentences(insideSentences)
      .setEntitiesPath("/entity-extractor/test-phrases.txt")
      .setOutputCol("entity")
    entityExtractor.transform(
      withFullNormalizer(
        withTokenizer(dataset)))
  }

  def withFullPragmaticSentenceDetector(dataset: Dataset[Row]): Dataset[Row] = {
    val sentenceDetector = new SentenceDetectorModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
    sentenceDetector.transform(dataset)
  }

  def withFullPOSTagger(dataset: Dataset[Row]): Dataset[Row] = {
    new PerceptronApproach()
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("pos")
      .fit(withFullPragmaticSentenceDetector(withTokenizer(dataset)))
      .transform(withFullPragmaticSentenceDetector(withTokenizer(dataset)))
  }

  def withRegexMatcher(dataset: Dataset[Row], rules: Array[(String, String)] = Array.empty[(String, String)], strategy: String): Dataset[Row] = {
    val regexMatcher = new RegexMatcher()
      .setStrategy(strategy)
      .setInputCols(Array("document"))
      .setOutputCol("regex")
    if (rules.nonEmpty) regexMatcher.setRules(rules)
    regexMatcher.transform(dataset)
  }

  def withDateMatcher(dataset: Dataset[Row]): Dataset[Row] = {
    val dateMatcher = new DateMatcher()
      .setOutputCol("date")
    dateMatcher.transform(dataset)
  }

  def withLemmaTaggedSentences(dataset: Dataset[Row]): Dataset[Row] = {
    withFullLemmatizer(withFullPOSTagger(dataset))
  }

  def withPragmaticSentimentDetector(dataset: Dataset[Row]): Dataset[Row] = {
    val sentimentDetector = new SentimentDetectorModel
    sentimentDetector
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("sentiment")
    sentimentDetector.transform(withFullPOSTagger(withFullLemmatizer(dataset)))
  }

  def withViveknSentimentAnalysis(dataset: Dataset[Row]): Dataset[Row] = {
    new ViveknSentimentApproach()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("vivekn")
      .setPositiveSourcePath("/vivekn/positive/1.txt")
      .setNegativeSourcePath("/vivekn/negative/1.txt")
      .setCorpusPrune(0)
      .fit(dataset)
      .transform(withTokenizer(dataset))
  }

  def withFullSpellChecker(dataset: Dataset[Row], inputFormat: String = "TXT"): Dataset[Row] = {
    val spellChecker = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")
      .setDictPath("./src/main/resources/spell/words.txt")
      .setCorpusPath("./src/test/resources/spell/sherlockholmes.txt")
      .setCorpusFormat(inputFormat)
    spellChecker.fit(withFullNormalizer(dataset)).transform(withFullNormalizer(dataset))
  }

  def withDependencyParser(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(withTokenizer(dataset))
    new DependencyParser()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setSourcePath("src/test/resources/models/dep-model.txt")
      .fit(df)
      .transform(df)
  }

  def withNerCrfTagger(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(withTokenizer(dataset))

    getNerCrfModel(dataset).transform(df)
  }

  def getNerCrfModel(dataset: Dataset[Row]): NerCrfModel = {
    val df = withFullPOSTagger(withTokenizer(dataset))

    new NerCrfApproach()
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setMinEpochs(1)
      .setMaxEpochs(3)
      .setDatasetPath("src/test/resources/ner-corpus/test_ner_dataset.txt")
      .setEmbeddingsSource("src/test/resources/ner-corpus/test_embeddings.txt", 3, WordEmbeddingsFormat.Text)
      .setC0(34)
      .setL2(3.0)
      .setOutputCol("ner")
      .fit(df)
  }
}

