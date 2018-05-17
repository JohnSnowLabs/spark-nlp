package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.ner.dl.{NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.Pipeline
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
    val tokenized = withTokenizer(dataset)
    normalizer.fit(dataset).transform(tokenized).show(5)
    normalizer.fit(dataset).transform(tokenized)
  }

  def withCaseSensitiveNormalizer(dataset: Dataset[Row]): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
      .setLowercase(false)
    val tokenized = withTokenizer(dataset)
    normalizer.fit(dataset).transform(tokenized).show(5)
    normalizer.fit(dataset).transform(tokenized)
  }

  def withFullLemmatizer(dataset: Dataset[Row]): Dataset[Row] = {
    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
      .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")
    val tokenized = withTokenizer(dataset)
    lemmatizer.fit(dataset).transform(tokenized)
  }

  def withFullTextMatcher(dataset: Dataset[Row], lowerCase: Boolean = true, sbd: Boolean = true): Dataset[Row] = {
    val entityExtractor = new TextMatcher()
      .setInputCols("normalized")
      .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.LINE_BY_LINE)
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

  var posTagger: PerceptronModel = null

  def withFullPOSTagger(dataset: Dataset[Row]): Dataset[Row] = {
    if (posTagger == null) {
     posTagger = new PerceptronApproach()
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("pos")
        .setNIterations(1)
        .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/110CYL067.txt", ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
        .fit(withFullPragmaticSentenceDetector(withTokenizer(dataset)))
    }

    posTagger.transform(withFullPragmaticSentenceDetector(withTokenizer(dataset)))
  }

  def withRegexMatcher(dataset: Dataset[Row], strategy: String): Dataset[Row] = {
    val regexMatcher = new RegexMatcher()
      .setRules(ExternalResource("src/test/resources/regex-matcher/rules.txt", ReadAs.LINE_BY_LINE, Map("delimiter" -> ",")))
      .setStrategy(strategy)
      .setInputCols(Array("document"))
      .setOutputCol("regex")
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
      .setDictionary(ExternalResource("src/test/resources/sentiment-corpus/default-sentiment-dict.txt", ReadAs.LINE_BY_LINE, Map("delimiter"->",")))
    sentimentDetector.fit(data).transform(data)
  }

  def withViveknSentimentAnalysis(dataset: Dataset[Row]): Dataset[Row] = {
    new ViveknSentimentApproach()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("vivekn")
      .setPositiveSource(ExternalResource("src/test/resources/vivekn/positive/1.txt", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "\\S+")))
      .setNegativeSource(ExternalResource("src/test/resources/vivekn/negative/1.txt", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "\\S+")))
      .setCorpusPrune(0)
      .fit(dataset)
      .transform(withTokenizer(dataset))
  }

  def withFullSpellChecker(dataset: Dataset[Row], inputFormat: String = "TXT"): Dataset[Row] = {
    val spellChecker = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")
      .setCorpus(ExternalResource("src/test/resources/spell/sherlockholmes.txt", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "\\S+")))
    spellChecker.fit(withFullNormalizer(dataset)).transform(withFullNormalizer(dataset))
  }

  def withDependencyParser(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(withTokenizer(dataset))
    new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setSource(ExternalResource("src/test/resources/models/dep-model.txt", ReadAs.LINE_BY_LINE, Map.empty[String, String]))
      .fit(df)
      .transform(df)
  }

  def withNerCrfTagger(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(dataset)

    getNerCrfModel(dataset).transform(df)
  }

  def getNerCrfModel(dataset: Dataset[Row]): NerCrfModel = {
    val df = withFullPOSTagger(dataset)

    new NerCrfApproach()
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setMinEpochs(1)
      .setMaxEpochs(3)
      .setEmbeddingsSource("src/test/resources/ner-corpus/test_embeddings.txt", 3, WordEmbeddingsFormat.TEXT)
      .setC0(34)
      .setL2(3.0)
      .setOutputCol("ner")
      .fit(df)
  }

  def withNerDLTagger(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(dataset)

    getNerDLModel(dataset).transform(df)
  }

  def getNerDLModel(dataset: Dataset[Row]): NerDLModel = {
    val df = withFullPOSTagger(dataset)

    new NerDLApproach()
      .setInputCols("sentence", "token")
      .setLabelColumn("label")
      .setMaxEpochs(100)
      .setRandomSeed(0)
      .setPo(0.01f)
      .setLr(0.1f)
      .setBatchSize(9)
      .setOutputCol("ner")
      .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt", 100, WordEmbeddingsFormat.TEXT)
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
    import java.io.{File, PrintWriter}

    import org.apache.spark.sql.functions._
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
      .setEmbeddingsSource("src/test/resources/random_embeddings_dim4.txt", 4, WordEmbeddingsFormat.TEXT)
      .setStartCol("start")
      .setEndCol("end")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, assertion)).fit(dataset)
    pipeline
  }
}

