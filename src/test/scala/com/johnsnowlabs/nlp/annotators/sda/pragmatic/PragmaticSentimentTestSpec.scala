/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.storage.StorageLevel
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.tagobjects.Slow

class PragmaticSentimentBigTestSpec extends AnyFlatSpec {

  "Parquet based data" should "be sentiment detected properly" taggedAs Slow in {
    import java.util.Date

    val data = ContentProvider.parquetData.limit(1000)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")

    val assembled = documentAssembler.transform(data)

    val sentimentDetector = new SentimentDetector()

    val readyData = AnnotatorBuilder.withFullPOSTagger(AnnotatorBuilder.withFullLemmatizer(assembled))
    
    val result = sentimentDetector
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("my_sda_scores")
      .setDictionary(ExternalResource("src/test/resources/sentiment-corpus/default-sentiment-dict.txt", ReadAs.TEXT, Map("delimiter" -> ",")))
      .setEnableScore(false)
      .fit(readyData)
      .transform(readyData)

    import Annotation.extractors._

    val date1 = new Date().getTime
    result.show(2)
    info(s"20 show sample of disk based sentiment analysis took: ${(new Date().getTime - date1)/1000} seconds")

    val date2 = new Date().getTime
    result.take("my_sda_scores", 5000)
    info(s"5000 take sample of disk based sentiment analysis took: ${(new Date().getTime - date2)/1000} seconds")

    val dataFromMemory = readyData.persist(StorageLevel.MEMORY_AND_DISK)
    info(s"data in memory is of size: ${dataFromMemory.count}")
    val resultFromMemory = sentimentDetector.fit(dataFromMemory).transform(dataFromMemory)

    val date3 = new Date().getTime
    resultFromMemory.show(2)
    info(s"20 show sample of memory based sentiment analysis took: ${(new Date().getTime - date3)/1000} seconds")

    val date4 = new Date().getTime
    resultFromMemory.take("my_sda_scores", 5000)
    info(s"5000 take sample of memory based sentiment analysis took: ${(new Date().getTime - date4)/1000} seconds")

    succeed
  }

}

class PragmaticSentimentTestSpec extends AnyFlatSpec with PragmaticSentimentBehaviors {

  val sentimentSentenceTexts = "The staff of the restaurant is nice and the eggplant is bad " +
    "I recommend others to avoid because it is too expensive"

  val sentimentSentences = {
    new Tokenizer().fit(ContentProvider.parquetData).tag(Sentence.fromTexts(sentimentSentenceTexts)).toArray
  }

  "an isolated sentiment detector" should behave like isolatedSentimentDetector(sentimentSentences, -4.0)

  "a spark based sentiment detector" should behave like sparkBasedSentimentDetector(
    DataBuilder.basicDataBuild("The staff of the restaurant is nice and the eggplant is bad." +
      " I recommend others to avoid.")
  )

  "A SentimentDetector" should "be readable and writable" taggedAs FastTest in {
    val sentimentDetector = new SentimentDetector().setDictionary(ExternalResource("src/test/resources/sentiment-corpus/default-sentiment-dict.txt", ReadAs.TEXT, Map("delimiter" -> ","))).fit(DataBuilder.basicDataBuild("dummy"))
    val path = "./test-output-tmp/sentimentdetector"
    try {
      sentimentDetector.write.overwrite.save(path)
      val sentimentDetectorRead = SentimentDetectorModel.read.load(path)
      assert(sentimentDetector.model.score(sentimentSentences) == sentimentDetectorRead.model.score(sentimentSentences))
    } catch {
      case _: java.io.IOException => succeed
    }
  }

}