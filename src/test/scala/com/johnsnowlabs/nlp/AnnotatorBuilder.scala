package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.ner.dl.{NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsFormat, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  * Generates different Annotator pipeline paths
  * Place to add different annotator constructions
  */
object AnnotatorBuilder extends FlatSpec { this: Suite =>

  def withDocumentAssembler(dataset: Dataset[Row], cleanupMode: String = "disabled"): Dataset[Row] = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setCleanupMode(cleanupMode)
    documentAssembler.transform(dataset)
  }

  def withTokenizer(dataset: Dataset[Row], sbd: Boolean = true): Dataset[Row] = {
    val regexTokenizer = new Tokenizer()
      .setOutputCol("token")
      .fit(dataset)
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
    normalizer.fit(dataset).transform(tokenized)
  }

  def withCaseSensitiveNormalizer(dataset: Dataset[Row]): Dataset[Row] = {
    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
      .setLowercase(false)
    val tokenized = withTokenizer(dataset)
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

  def withFullTextMatcher(dataset: Dataset[Row], caseSensitive: Boolean = true, sbd: Boolean = true): Dataset[Row] = {
    val entityExtractor = new TextMatcher()
      .setInputCols(if (sbd) "sentence" else "document", "token")
      .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.LINE_BY_LINE)
      .setOutputCol("entity").
      setCaseSensitive(caseSensitive)
    val data = withTokenizer(dataset, sbd)
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

      val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/110CYL067.txt", "|", "tags")

      posTagger = new PerceptronApproach()
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("pos")
        .setPosColumn("tags")
        .setNIterations(1)
        .fit(trainingPerceptronDF)
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

  def withMultiDateMatcher(dataset: Dataset[Row]): Dataset[Row] = {
    val dateMatcher = new MultiDateMatcher()
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

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val sentimentDetector = new ViveknSentimentApproach()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("vivekn")
      .setSentimentCol("sentiment_label")
      .setCorpusPrune(0)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        sentimentDetector
      ))

    pipeline.fit(dataset).transform(dataset)

  }

  def withFullSpellChecker(dataSet: Dataset[Row], inputFormat: String = "TXT"): Dataset[Row] = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        normalizer,
        spellChecker
      ))
    val trainingDataSet = getTrainingDataSet("src/test/resources/spell/sherlockholmes.txt")
    val predictionDataSet = withFullNormalizer(dataSet)
    val model = pipeline.fit(trainingDataSet)
    model.transform(predictionDataSet)
  }

  def withTreeBankDependencyParser(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(withTokenizer(dataset))
    new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
      .fit(df)
      .transform(df)
  }

  def withNerCrfTagger(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withGlove(dataset)

    getNerCrfModel(dataset).transform(df)
  }

  def getNerCrfModel(dataset: Dataset[Row]): NerCrfModel = {
    val df = withGlove(dataset)

    new NerCrfApproach()
      .setInputCols("sentence", "token", "pos", "embeddings")
      .setLabelColumn("label")
      .setMinEpochs(1)
      .setMaxEpochs(3)
      .setC0(34)
      .setL2(3.0)
      .setOutputCol("ner")
      .fit(df)
  }

  def withGlove(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withFullPOSTagger(dataset)

    getGLoveEmbeddings(dataset).transform(df)
  }

  def getGLoveEmbeddings(dataset: Dataset[Row]): WordEmbeddingsModel = {
    new WordEmbeddings()
      .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt", 100, WordEmbeddingsFormat.TEXT)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .fit(dataset)
  }

  def withNerDLTagger(dataset: Dataset[Row]): Dataset[Row] = {
    val df = withGlove(dataset)

    getNerDLModel(dataset).transform(df)
  }

  def getNerDLModel(dataset: Dataset[Row]): NerDLModel = {
    val df = withGlove(dataset)

    new NerDLApproach()
      .setInputCols("sentence", "token", "embeddings")
      .setLabelColumn("label")
      .setMaxEpochs(100)
      .setRandomSeed(0)
      .setPo(0.01f)
      .setLr(0.1f)
      .setBatchSize(9)
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

  def getTrainingDataSet(textFile: String): Dataset[_] = {
    val trainingDataSet = SparkAccessor.spark.read.textFile(textFile)
    trainingDataSet.select(trainingDataSet.col("value").as("text"))
  }

}

