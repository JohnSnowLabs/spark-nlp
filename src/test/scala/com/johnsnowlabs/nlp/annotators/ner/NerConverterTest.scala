package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.scalatest._

class NerConverterTest extends FlatSpec {

  "NeConverter" should "correctly use any TOKEN type input" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/ner-corpus/test_ner_dataset.txt")

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(false)

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalize = new Normalizer()
      .setInputCols("token")
      .setOutputCol("cleaned")
      .setLowercase(true)

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("document", "cleaned")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val ner = NerDLModel.pretrained()
      .setInputCols("sentence", "cleaned", "embeddings")
      .setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("sentence", "cleaned", "ner")
      .setOutputCol("entities")
      .setPreservePosition(false)

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        normalize,
        embeddings,
        ner,
        converter
      ))

    val nermodel = recursivePipeline.fit(training_data).transform(training_data)

    nermodel.select("token.result").show(1, false)
    nermodel.select("cleaned.result").show(1, false)
    nermodel.select("embeddings.result").show(1, false)
    nermodel.select("entities.result").show(1, false)
    nermodel.select("entities").show(1, false)

  }

}

