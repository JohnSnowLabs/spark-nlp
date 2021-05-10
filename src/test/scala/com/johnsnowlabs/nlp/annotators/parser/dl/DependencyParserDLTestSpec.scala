package com.johnsnowlabs.nlp.annotators.parser.dl

import com.johnsnowlabs.nlp.training.CoNLLU
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.FlatSpec

class DependencyParserDLTestSpec extends FlatSpec {

  val conlluFile = "src/test/resources/conllu/en.test.conllu"

  "Dependency Parser DL" should "load trainable variables" in {
    val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)
    conllDataSet.show(5, false)

    val dependencyParser = new DependencyParserDLApproach()
      .setInputCols("document")
      .setOutputCol("dependency")

    dependencyParser.fit(conllDataSet)

  }

  it should "get weights on model" in {
    //ConfigHelper.hasPath("sparknlp.settings.useBroadcastForFeatures")
    val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)
    conllDataSet.show(5, false)

    val dependencyParser = new DependencyParserDLApproach()
      .setInputCols("document", "lemma")
      .setOutputCol("dependency")

    dependencyParser.fit(conllDataSet).transform(conllDataSet).show(false)
  }

}
