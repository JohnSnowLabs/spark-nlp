package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.assertion.logreg.{AssertionLogRegApproach, AssertionLogRegModel}
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.apache.spark.ml.{Pipeline, PipelineModel}
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

  def withTokenizer(dataset: Dataset[Row], sbd: Boolean = true): Dataset[Row] = {
    val regexTokenizer = new Tokenizer()
      .setOutputCol("token")
    if (sbd)
      regexTokenizer.setInputCols(Array("sentence")).transform(withFullPragmaticSentenceDetector(dataset))
    else
      regexTokenizer.setInputCols(Array("document")).transform(dataset)
  }

  def withFullStemmer(dataset: Dataset[Row]): Dataset[Row] = {
    val stemmer = new Stemmer()
      .setInputCols(Array("token"))
      .setOutputCol("stem")
    stemmer.transform(withTokenizer(dataset))
  }

  def withFullNormalizer(dataset: Dataset[Row], lowerCase: Boolean = true): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
      .setLowercase(lowerCase)
    normalizer.transform(withTokenizer(dataset))
  }

  def withCaseSensitiveNormalizer(dataset: Dataset[Row]): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
      .setLowercase(false)
    normalizer.transform(withTokenizer(dataset))
  }

  def withFullLemmatizer(dataset: Dataset[Row]): Dataset[Row] = {
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
      .setLemmaDictPath("/lemma-corpus/AntBNC_lemmas_ver_001.txt")
    val tokenized = withTokenizer(dataset)
    lemmatizer.fit(dataset).transform(tokenized)
  }

  def withFullEntityExtractor(dataset: Dataset[Row], lowerCase: Boolean = true, sbd: Boolean = true): Dataset[Row] = {
    val entityExtractor = new EntityExtractor()
      .setInputCols("normalized")
      .setEntitiesPath("/entity-extractor/test-phrases.txt")
      .setOutputCol("entity")
    val data = withFullNormalizer(
      withTokenizer(dataset, sbd), lowerCase)
    entityExtractor.fit(data).transform(data)
  }

  def withFullPragmaticSentenceDetector(dataset: Dataset[Row]): Dataset[Row] = {
    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
    sentenceDetector.transform(dataset)
  }

  def withFullPOSTagger(dataset: Dataset[Row]): Dataset[Row] = {
    new PerceptronApproach()
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("pos")
      .setCorpusPath("/anc-pos-corpus-small/")
      .fit(withFullPragmaticSentenceDetector(withTokenizer(dataset)))
      .transform(withFullPragmaticSentenceDetector(withTokenizer(dataset)))
  }

  def withRegexMatcher(dataset: Dataset[Row], rules: Array[(String, String)] = Array.empty[(String, String)], strategy: String): Dataset[Row] = {
    val regexMatcher = new RegexMatcher()
      .setRulesPath("/regex-matcher/rules.txt")
      .setStrategy(strategy)
      .setInputCols(Array("document"))
      .setOutputCol("regex")
    if (rules.nonEmpty) regexMatcher.setRules(rules)
    regexMatcher.fit(dataset).transform(dataset)
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
    val data = withFullPOSTagger(withFullLemmatizer(dataset))
    val sentimentDetector = new SentimentDetector
    sentimentDetector
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("sentiment")
      .setDictPath("/sentiment-corpus/default-sentiment-dict.txt")
    sentimentDetector.fit(data).transform(data)
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
    new DependencyParserApproach()
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

  /* generate a set of random embeddings from tokens in dataset
  *  rowText is the column containing the text.
  *  returns the path of the file
  *
  *  usage,
  *  val embeddingsPath = generateRandomEmbeddings(dataset, "sentence", 4)
  * */
  private def generateRandomEmbeddings(dataset: Dataset[Row], rowText: String, dim: Int) = {
    import org.apache.spark.sql.functions._
    import java.io.{PrintWriter, File}
    val random = scala.util.Random
    val filename = s"${rowText}_${dim}.txt"
    val pw = new PrintWriter(new File(filename))

    val tokens = dataset.toDF().select(col(rowText)).
      collect().flatMap(row=> row.getString(0).split(" ")).
      distinct

    def randomDoubleArrayStr = (1 to dim).map{_ => random.nextDouble}.mkString(" ")

    for (token <- tokens)
      pw.println(s"$token $randomDoubleArrayStr")

    filename
  }

  def getAssertionLogregModel(dataset: Dataset[Row]) = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("sentence")
      .setOutputCol("document")

    val assertion = new AssertionLogRegApproach()
      .setLabelCol("label")
      .setInputCols("document")
      .setOutputCol("assertion")
      .setReg(0.01)
      .setBefore(11)
      .setAfter(13)
      .setEmbeddingsSource("src/test/resources/random_embeddings_dim4.txt", 4, WordEmbeddingsFormat.Text)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, assertion)).fit(dataset)
    pipeline
  }
}

