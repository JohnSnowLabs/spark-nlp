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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.{Benchmark, FileHelper}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class NerDLSpec extends AnyFlatSpec {

  "NER DL Approach" should "train and annotate with perf" taggedAs SlowTest in {

    val smallCustomDataset = "src/test/resources/conll2003/eng.testa"
    SparkAccessor.spark.conf.getAll.foreach(println)

    for (samples <- Array(512, 1024, 2048, 4096, 8192)) {

      val trainData =
        CoNLL(conllLabelIndex = 1)
          .readDatasetFromLines(
            Source.fromFile(smallCustomDataset).getLines.toArray,
            SparkAccessor.spark)
          .toDF
          .limit(samples)
      println(s"TRAIN DATASET COUNT: ${trainData.count}")

      val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
      val sentence =
        new SentenceDetector().setInputCols(Array("document")).setOutputCol("sentence")
      val token = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")
      val bert = BertEmbeddings
        .pretrained("bert_base_cased", "en")
        .setInputCols(Array("sentence", "token"))
        .setOutputCol("bert")

      val testDataset = trainData
      println(s"TEST DATASET COUNT: ${testDataset.count}")

      val testDataParquetPath = "src/test/resources/conll2003/testDataParquet"
      bert.transform(testDataset).write.mode("overwrite").parquet(testDataParquetPath)

      val nerTagger = new NerDLApproach()
        .setInputCols(Array("sentence", "token", "bert"))
        .setLabelColumn("label")
        .setOutputCol("ner")
        .setMaxEpochs(10)
        .setLr(0.001f)
        .setPo(0.005f)
        .setBatchSize(8)
        .setRandomSeed(0)
        .setVerbose(0)
        .setValidationSplit(0.2f)
        .setEvaluationLogExtended(false)
        .setEnableOutputLogs(false)
        .setIncludeConfidence(true)
        .setTestDataset(testDataParquetPath)

      val pipeline =
        new Pipeline()
          .setStages(Array(document, sentence, token, bert, nerTagger))

      val fitted =
        Benchmark.time(s"$samples fit ner dl time", forcePrint = true) {
          pipeline.fit(trainData)
        }
      val transformed =
        Benchmark.time(s"$samples transform ner dl time", forcePrint = true) {
          fitted.transform(testDataset)
        }

      println(s"transformed.count: ${transformed.count}")
      transformed.write
        .mode("overwrite")
        .parquet(s"src/test/resources/conll2003/out/transformedParquet${samples}")
    }
  }

  "NerDLApproach" should "correctly annotate" taggedAs SlowTest in {
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    //    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    //    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)

    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten.toSeq
    val labels = Annotation.collect(tagged, "label").flatten.toSeq

    assert(annotations.length == labels.length)
    for ((annotation, label) <- annotations.zip(labels)) {
      assert(annotation.begin == label.begin)
      assert(annotation.end == label.end)
      assert(annotation.annotatorType == AnnotatorType.NAMED_ENTITY)
      assert(annotation.result == label.result)
      assert(annotation.metadata.contains("word"))
    }
  }

  "NerDLApproach" should "correctly tag sentences" taggedAs SlowTest in {
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)

    val tagged = nerModel.transform(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLModel" should "correctly train using dataset from file" taggedAs SlowTest in {
    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val tagged = AnnotatorBuilder.withNerDLTagger(nerInputDataset)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLApproach" should "be serializable and deserializable correctly" taggedAs SlowTest in {

    val nerSentence = DataBuilder.buildNerDataset(ContentProvider.nerCorpus)
    System.out.println(s"number of sentences in dataset ${nerSentence.count()}")

    // Dataset ready for NER tagger
    val nerInputDataset = AnnotatorBuilder.withGlove(nerSentence)
    System.out.println(s"number of sentences in dataset ${nerInputDataset.count()}")

    val nerModel = AnnotatorBuilder.getNerDLModel(nerSentence)

    nerModel.write.overwrite.save("./test_ner_dl")
    val loadedNer = NerDLModel.read.load("./test_ner_dl")
    FileHelper.delete("./test_ner_dl")

    // Test that params of loaded model are the same
    assert(loadedNer.datasetParams.getOrDefault == nerModel.datasetParams.getOrDefault)

    // Test that loaded model do the same predictions
    val tokenized = AnnotatorBuilder.withTokenizer(nerInputDataset)
    val tagged = loadedNer.transform(tokenized)
    val annotations = Annotation.collect(tagged, "ner").flatten

    val tags = annotations.map(a => a.result).toSeq
    assert(tags.toList == Seq("PER", "PER", "O", "O", "ORG", "LOC", "O"))
  }

  "NerDLApproach" should "correct search for suitable graphs" taggedAs FastTest in {
    val smallGraphFile = NerDLApproach.searchForSuitableGraph(10, 100, 120)
    assert(smallGraphFile.endsWith("blstm_10_100_128_120.pb"))

    val bigGraphFile = NerDLApproach.searchForSuitableGraph(25, 300, 120)
    assert(bigGraphFile.endsWith("blstm_38_300_128_200.pb"))

    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(31, 101, 100))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(50, 300, 601))
    assertThrows[IllegalArgumentException](NerDLApproach.searchForSuitableGraph(40, 512, 101))
  }

  "NerDLApproach" should "validate against part of the training dataset" taggedAs FastTest in {

    val conll = CoNLL()
    val training_data = conll.readDataset(
      ResourceHelper.spark,
      "src/test/resources/ner-corpus/test_ner_dataset.txt")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(training_data.toDF())

    val trainData = embeddings.transform(training_data)

    val ner = new NerDLApproach()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setLr(1e-1f) // 0.1
      .setPo(5e-3f) // 0.005
      .setDropout(5e-1f) // 0.5
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setVerbose(0)
      .setEvaluationLogExtended(true)
      .setEnableOutputLogs(true)
      .setGraphFolder("src/test/resources/graph/")
      .setUseBestModel(true)
      .fit(trainData)

    ner.write.overwrite() save ("./tmp_ner_dl_tf115")
  }

  "NerDLModel" should "successfully load saved model" taggedAs FastTest in {

    val conll = CoNLL()
    val test_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testb")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(test_data.toDF())

    val testData = embeddings.transform(test_data)

    NerDLModel
      .load("./tmp_ner_dl_tf115")
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .transform(testData)

  }

  "NerDLModel" should "successfully download a pretrained model" taggedAs FastTest in {

    val nerModel = NerDLModel
      .pretrained()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")

    nerModel.getClasses.foreach(x => println(x))

  }

  "NerDLApproach" should "benchmark test" taggedAs SlowTest in {

    val conll = CoNLL(explodeSentences = false)
    val trainingData =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")
    val testingData =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.testa")

    val embeddings = WordEmbeddingsModel.pretrained()

    val trainDF = embeddings.transform(trainingData)
    embeddings.transform(testingData).write.mode("overwrite").parquet("./tmp_test_coll")

    val nerModel = new NerDLApproach()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setLr(1e-3f) // 0.001
      .setPo(5e-3f) // 0.005
      .setDropout(5e-1f) // 0.5
      .setMaxEpochs(5)
      .setRandomSeed(0)
      .setVerbose(0)
      .setBatchSize(8)
      .setEvaluationLogExtended(true)
      .setGraphFolder("src/test/resources/graph/")
      .setTestDataset("./tmp_test_coll")
      .setUseBestModel(true)
      .fit(trainDF)

    nerModel.write.overwrite() save ("./tmp_ner_dl_glove_conll03_100d")
  }

  "NerDLModel" should "benchmark test" taggedAs SlowTest in {

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = WordEmbeddingsModel.pretrained()

    val nerModel = NerDLModel
      .pretrained()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .setBatchSize(8)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings, nerModel))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)

    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.select("ner.result").write.mode("overwrite").parquet("./tmp_nerdl")
    }
  }

  "NerDLModel" should "work with confidence scores enabled" taggedAs SlowTest in {

    val conll = CoNLL(explodeSentences = false)
    val training_data =
      conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val embeddings = WordEmbeddingsModel.pretrained()

    val nerModel = NerDLModel
      .pretrained()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")
      .setIncludeConfidence(true)

    val pipeline = new Pipeline()
      .setStages(Array(embeddings, nerModel))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    pipelineDF.select("ner").show(1, truncate = false)
  }

  // AWS keys need to be set up for this test
  ignore should "correct search for suitable graphs on S3" taggedAs SlowTest in {
    val awsAccessKeyId = sys.env("AWS_ACCESS_KEY_ID")
    val awsSecretAccessKey = sys.env("AWS_SECRET_ACCESS_KEY")
    val awsSessionToken = sys.env("AWS_SESSION_TOKEN")

    ResourceHelper.getSparkSessionWithS3(awsAccessKeyId, awsSecretAccessKey, awsSessionToken)

    val s3FolderPath = "s3://devin-sparknlp-test/ner-dl/"  // identical to the one in repository
    val smallGraphFile = NerDLApproach.searchForSuitableGraph(10, 100, 120, Some(s3FolderPath))
    assert(smallGraphFile.endsWith("blstm_10_100_128_120.pb"))

    val bigGraphFile = NerDLApproach.searchForSuitableGraph(25, 300, 120, Some(s3FolderPath))
    assert(bigGraphFile.endsWith("blstm_38_300_128_200.pb"))

  }

}
