package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.{NerDLApproach, WordEmbeddings}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.scalatest.FlatSpec

class NerDLEvalTesSpec extends FlatSpec {

  "A NER DL Evaluation" should "display accuracy results" in {
    val trainFile = "./eng.train"
    val testFile = "./eng.testa"
    val format = ""
    val modelPath = "./ner_dl"

    val glove = new WordEmbeddings()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")
      .setEmbeddingsSource("./glove.6B.100d.txt ",
        100, WordEmbeddingsFormat.TEXT)
      .setCaseSensitive(true)

    val nerTagger = new NerDLApproach()
      .setInputCols(Array("sentence", "token", "glove"))
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setRandomSeed(0)
      .setVerbose(2)

    val nerDLEvaluation = new NerDLEvaluation(testFile, format)
    nerDLEvaluation.computeAccuracyAnnotator(modelPath, trainFile, nerTagger, glove)

  }

}