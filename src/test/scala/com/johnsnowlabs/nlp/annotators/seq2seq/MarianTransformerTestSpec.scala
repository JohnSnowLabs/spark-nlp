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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

class MarianTransformerTestSpec extends AnyFlatSpec {

  "MarianTransformer" should "load saved model from onnx" taggedAs SlowTest in {
    MarianTransformer
      .loadSavedModel("/tmp/mt_en_bg", ResourceHelper.spark)
      .save("/models/sparknlp/marianmt_onnx")
  }
  "MarianTransformer" should "do some work with ONNX" in {

    import ResourceHelper.spark.implicits._

    val smallCorpus = Seq(
      "Which is the capital of France?",
      "This should go to French.",
      "This is a sentence in English which we want to translate to French.",
      "Despite a Democratic majority in the General Assembly, Nunn was able to enact most of his priorities, including tax increases that funded improvements to the state park system and the construction of a statewide network of mental health centers.",
      "",
      " ").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "xx")
      .setInputCols("document")
      .setOutputCol("sentence")

    val marian = MarianTransformer
      .load("/models/sparknlp/marianmt_onnx")
//        .pretrained("opus_mt_en_bg", "xx")
      .setInputCols("sentence")
      .setOutputCol("translation")
      .setMaxInputLength(512)
      .setMaxOutputLength(128)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, marian))

    val pipelineModel = pipeline.fit(smallCorpus)
    pipelineModel
      .transform(smallCorpus)
      .select(
        col("sentence.result").alias("source"),
        col("translation.result").alias("translation"))
      .show(truncate = false)
  }

  "MarianTransformer" should "ignore bad token ids" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark
      .createDataFrame(Seq(
        (
          1,
          "Le principal facteur de réchauffement est l'émission de gaz à effet de serre, dont plus de 90 % sont le dioxyde de " +
            "carbone (CO2) et le méthane. La combustion de combustibles fossiles comme le charbon, le pétrole et le gaz naturel pour " +
            "la consommation d'énergie est la principale source de ces émissions, avec des contributions supplémentaires de l'agriculture, " +
            "de la déforestation et de la production industrielle. La cause humaine du changement climatique n'est contestée par aucun organisme " +
            "scientifique de renommée nationale ou internationale. L'augmentation de la température est accélérée ou tempérée par les rétroactions " +
            "climatiques, telles que la perte de couverture de neige et de glace réfléchissant la lumière du soleil, l'augmentation de la vapeur " +
            "d'eau (un gaz à effet de serre lui-même) et les modifications des puits de carbone terrestres et océaniques."),
        (
          1,
          "Donald John Trump (pronunciación en inglés: /ˈdɒnəld d͡ʒɒn trʌmp/ ( escuchar); Nueva York, 14 de junio de 1946) " +
            "es un empresario, director ejecutivo, inversor en bienes inmuebles, personalidad televisiva y político estadounidense " +
            "que ejerció como el 45.º presidente de los Estados Unidos de América desde el 20 de enero de 2017 hasta el 20 de enero de 2021.2 " +
            "Nacido y criado en un barrio del distrito neoyorquino de Queens llamado «Jamaica», Trump obtuvo el título de bachiller en economía en " +
            "la Wharton School de la Universidad de Pensilvania en 1968. En 1971, se hizo cargo de la empresa familiar de bienes inmuebles y construcción " +
            "Elizabeth Trump & Son, que más tarde sería renombrada como Trump Organization. Durante su carrera empresarial, Trump construyó, renovó y " +
            "gestionó numerosas torres de oficinas, hoteles, casinos y campos de golf. Fue accionista principal de los concursos de belleza Miss USA y " +
            "Miss Universo desde 1996 hasta 2015, y prestó el uso de su nombre en la marca de varios productos. De 2004 a 2015, participó en The Apprentice, " +
            "un reality show de la NBC. En 2016, la revista Forbes lo enumeró como la 324.ª persona más rica del mundo (la 113.ª de los Estados Unidos), con un " +
            "valor neto de 4500 millones de dólares. Según las estimaciones de Forbes en febrero de 2018, Trump se encuentra entre las personas más ricas" +
            " del mundo en el puesto 766, con un valor neto de 3100 millones de dólares."),
        (2, "Това е български език."),
        (3, "Y esto al español."),
        (4, "Isto deve ir para o português.")))
      .toDF("id", "text")
      .repartition(1)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "xx")
      .setInputCols("document")
      .setOutputCol("sentence")

    val marian = MarianTransformer
      .pretrained("opus_mt_mul_en", "xx")
      .setInputCols("sentence")
      .setOutputCol("translation")
      .setBatchSize(1)
      .setMaxInputLength(50)
      .setIgnoreTokenIds(Array(64171))

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, marian))

    val pipelineModel = pipeline.fit(smallCorpus)
    val results = pipelineModel.transform(smallCorpus)

    Benchmark.time("Time to save pipeline the first time") {
      results.select("translation.result").write.mode("overwrite").save("./tmp_t5_pipeline")
    }

    Benchmark.time("Time to save pipeline the second time") {
      results.select("translation.result").write.mode("overwrite").save("./tmp_t5_pipeline")
    }

    Benchmark.time("Time to show") {
      val results = pipelineModel
        .transform(smallCorpus)
        .selectExpr("explode(translation) as translation")
        .where("length(translation.result) > 0")
        .selectExpr("translation.result as translation")
      assert(results.count() > 0, "Should return non-empty translations")
      results.show(truncate = false)
    }
  }

  "MarianTransformer" should "correctly load pretrained model" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val smallCorpus = Seq(
      "What is the capital of France?",
      "This should go to French",
      "This is a sentence in English that we want to translate to French",
      "Despite a Democratic majority in the General Assembly, Nunn was able to enact most of his priorities, including tax increases that funded improvements to the state park system and the construction of a statewide network of mental health centers.",
      "",
      " ").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "xx")
      .setInputCols("document")
      .setOutputCol("sentence")

    val marian = MarianTransformer
      .pretrained()
      .setInputCols("document")
      .setOutputCol("translation")
      .setMaxInputLength(512)
      .setMaxOutputLength(50)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, marian))

    val pipelineModel = pipeline.fit(smallCorpus)

    Benchmark.time("Time to save pipeline the first time") {
      pipelineModel
        .transform(smallCorpus)
        .select("translation.result")
        .write
        .mode("overwrite")
        .save("./tmp_marianmt_pipeline")
    }

    Benchmark.time("Time to save pipeline the second time") {
      pipelineModel
        .transform(smallCorpus)
        .select("translation.result")
        .write
        .mode("overwrite")
        .save("./tmp_marianmt_pipeline")
    }

    Benchmark.time("Time to first show") {
      pipelineModel.transform(smallCorpus).select("translation").show(false)
    }

    Benchmark.time("Time to second show") {
      pipelineModel.transform(smallCorpus).select("translation").show(false)
    }

    Benchmark.time("Time to save pipelineMolde") {
      pipelineModel.write.overwrite.save("./tmp_marianmt")
    }

    val savedPipelineModel = Benchmark.time("Time to load pipelineMolde") {
      PipelineModel.load("./tmp_marianmt")
    }
    val pipelineDF = Benchmark.time("Time to transform") {
      savedPipelineModel.transform(smallCorpus)
    }

    Benchmark.time("Time to show") {
      pipelineDF.select("translation").show(false)
    }
    Benchmark.time("Time to second show") {
      pipelineDF.select("translation").show(false)
    }

  }

}
