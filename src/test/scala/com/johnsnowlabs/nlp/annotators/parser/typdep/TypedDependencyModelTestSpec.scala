package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.{DataBuilder, DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class TypedDependencyModelTestSpec extends FlatSpec {

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  private val posTagger = getPerceptronModel

  private val dependencyParser = DependencyParserModel.read.load("./tmp/dp_model")

  private val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("token", "pos", "dependency"))
    .setOutputCol("labdep")
    .setConll2009FilePath("src/test/resources/parser/train/example.train")

  private val emptyDataset = PipelineModels.dummyDataset

  def getPerceptronModel: PerceptronModel = {
    val perceptronTagger = new PerceptronApproach()
      .setNIterations(1)
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/",
        ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("pos")
      .fit(DataBuilder.basicDataBuild("dummy"))
    val path = "./test-output-tmp/perceptrontagger"

    perceptronTagger.write.overwrite.save(path)
    val perceptronTaggerRead = PerceptronModel.read.load(path)
    perceptronTaggerRead
  }

  "A typed dependency parser model with a sentence input" should
    "predict a labeled relationship between words in the sentence" in {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataset)

    val sentence =
      "The most troublesome report may be the August merchandise trade deficit due out tomorrow."
    val helloDataset = Seq(sentence).toDS.toDF("text")
    val result = model.transform(helloDataset)
    result.show()

  }


  "A typed dependency parser model with a document input" should
    "predict a labeled relationship between words in each sentence" ignore {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataset)

    val document = "The most troublesome report may be the August merchandise trade deficit due out tomorrow. " +
                   "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward."
    val helloDataset = Seq(document).toDS.toDF("text")
    //helloDataset.show(1, false)
    val result = model.transform(helloDataset)
    result.show()

  }

}
