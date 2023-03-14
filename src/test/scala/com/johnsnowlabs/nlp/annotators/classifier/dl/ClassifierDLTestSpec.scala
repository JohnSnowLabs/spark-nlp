/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class ClassifierDLTestSpec extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  "ClassifierDL" should "correctly train IMDB train dataset" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/classifier/sentiment.csv")

    println("count of training dataset: ", smallCorpus.count)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceEmbeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(8)
      .setMaxEpochs(1)
      .setLr(5e-3f)
      .setDropout(0.5f)
      .setRandomSeed(44)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceEmbeddings, docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)

    pipelineModel.transform(smallCorpus).select("document").show(1, truncate = false)

  }

  "ClassifierDL" should "not fail on empty inputs" taggedAs SlowTest in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (1, "This is my first sentence. This is my second."),
          (2, "This is my third sentence. . . . .... ..."),
          (3, "")))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val sarcasmDL = ClassifierDLModel
      .pretrained(name = "classifierdl_use_sarcasm")
      .setInputCols("sentence_embeddings")
      .setOutputCol("sarcasm")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, useEmbeddings, sarcasmDL))

    val pipelineDF = pipeline.fit(testData).transform(testData)
    pipelineDF.select("sentence.result").show(false)
    pipelineDF.select("sentence_embeddings.result").show(false)
    pipelineDF.select("sarcasm.result").show(false)

    pipelineDF.show()

  }

  "ClassifierDL" should "correctly download and load pre-trained model" taggedAs FastTest in {
    val classifierDL = ClassifierDLModel.pretrained("classifierdl_use_trec50")
    classifierDL.getClasses.foreach(x => print(x + ", "))
  }

  "ClassifierDL" should "not fail on empty validation sets" taggedAs SlowTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceEmbeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(8)
      .setMaxEpochs(1)
      .setLr(5e-3f)
      .setDropout(0.5f)
      .setValidationSplit(0.1f)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceEmbeddings, docClassifier))

    val data = Seq(("This is good.", "good"), ("This is bad.", "bad"), ("This has no labels", ""))
      .toDF("text", "label")

    val pipelineModel = pipeline.fit(data)

    pipelineModel.transform(data).select("document").show(1, truncate = false)
  }

}
