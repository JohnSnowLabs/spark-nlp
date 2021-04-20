/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.benchmarks.spark

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.ner.dl.{NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.ner.{NerConverter, Verbose}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.PipelineModel


object NerDLCoNLL2003 extends App {
  val folder = "src/test/resources/conll2003/"

  val trainFile = ExternalResource(folder + "eng.train", ReadAs.TEXT, Map.empty[String, String])
  val testFileA = ExternalResource(folder + "eng.testa", ReadAs.TEXT, Map.empty[String, String])
  val testFileB = ExternalResource(folder + "eng.testb", ReadAs.TEXT, Map.empty[String, String])

  val nerReader = CoNLL()

  def createPipeline() = {

    val glove = WordEmbeddingsModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("glove")

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token", "glove")
      .setLabelColumn("label")
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setPo(0.005f)
      .setLr(1e-3f)
      .setDropout(0.5f)
      .setBatchSize(32)
      .setOutputCol("ner")
      .setVerbose(Verbose.Epochs)
      .setGraphFolder("src/test/resources/graph/")

    val converter = new NerConverter()
      .setInputCols("document", "token", "ner")
      .setOutputCol("ner_span")

    val labelConverter = new NerConverter()
      .setInputCols("document", "token", "label")
      .setOutputCol("label_span")

    Array(
      glove,
      nerTagger,
      converter,
      labelConverter
    )
  }

  def trainNerModel(er: ExternalResource): PipelineModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = nerReader.readDataset(SparkAccessor.benchmarkSpark, er.path)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val stages = createPipeline()

    val pipeline = new RecursivePipeline()
      .setStages(stages)

    pipeline.fit(dataset)
  }

  def getUserFriendly(model: PipelineModel, file: ExternalResource): Array[Array[Annotation]] = {
    val df = model.transform(nerReader.readDataset(SparkAccessor.benchmarkSpark, file.path))
    Annotation.collect(df, "ner_span")
  }

  def measure(model: PipelineModel, file: ExternalResource, extended: Boolean = true, errorsToPrint: Int = 0): Unit = {
    val ner = model.stages.filter(s => s.isInstanceOf[NerDLModel]).head.asInstanceOf[NerDLModel].getModelIfNotSet
    val df = nerReader.readDataset(SparkAccessor.benchmarkSpark, file.path).toDF()
    val transformed = model.transform(df)

    val labeled = NerTagged.iterateOnArray(transformed.collect(), Seq("sentence", "token", "glove"), 2)

    ner.measure(labeled, extended, outputLogsPath = "")
  }

  val spark = SparkAccessor.benchmarkSpark

  val model = trainNerModel(trainFile)

  measure(model, trainFile, false)
  measure(model, testFileA, false)
  measure(model, testFileB, true)

  val annotations = getUserFriendly(model, testFileB)
  NerHelper.saveNerSpanTags(annotations, "predicted.csv")

  model.write.overwrite().save("ner_model")
  PipelineModel.read.load("ner_model")

  System.out.println("Training dataset")
  NerHelper.measureExact(nerReader, model, trainFile)

  System.out.println("Validation dataset")
  NerHelper.measureExact(nerReader, model, testFileA)

  System.out.println("Test dataset")
  NerHelper.measureExact(nerReader, model, testFileB)
}
