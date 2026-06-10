/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.sbd.sat

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Integration tests for [[SentenceDetectorSaTModel]].
  *
  * These require a local ONNX export of a SaT model (e.g. `segment-any-text/sat-12l-sm`) laid out
  * as `model.onnx` + `assets/sentencepiece.bpe.model`. The export is expected in the folder `"1"`
  * at the repository root, so the tests are tagged [[SlowTest]].
  */
class SentenceDetectorSaTSpec extends AnyFlatSpec with Matchers {

  implicit val spark: SparkSession = ResourceHelper.spark

  /** Load the model from the local export folder and wire up the standard document column. */
  private def loadModel(): SentenceDetectorSaTModel =
    SentenceDetectorSaTModel
      .loadSavedModel("1", spark)
      .setInputCols("document")
      .setOutputCol("sentence")

  private val documentAssembler =
    new DocumentAssembler().setInputCol("text").setOutputCol("document")

  /** Run the model over a single text and return the detected sentence strings. */
  private def segment(text: String, model: SentenceDetectorSaTModel): Array[String] = {
    import spark.implicits._
    val data = Seq(text).toDF("text")
    val result = new Pipeline()
      .setStages(Array(documentAssembler, model))
      .fit(data)
      .transform(data)
    result.show(truncate = false)
    result.selectExpr("explode(sentence.result) as s").as[String].collect()
  }

  // 1. Basic English segmentation: a three-sentence paragraph should yield multiple sentences.
  "SentenceDetectorSaTModel" should "segment English sentences" taggedAs SlowTest in {
    val sentences = segment(
      "ontem fui ao hospital com minha mae ela estava se sentindo mal desde sexta passada "+
    "ficamos esperando tres horas na triagem o pessoal da recepcao era simpatico mas "+
    "estava claramente sobrecarregado tinha gente esperando desde a madrugada alguns "+
    "deitados nas cadeiras outros no chao com cobertores de hospital minha mae tem "+
    "sessenta e oito anos e pressao alta e fiquei preocupada porque ela estava palida "+
    "e suando frio quando finalmente chamaram a medica foi muito atenciosa fez todos "+
    "os exames pediu eletrocardiograma raio x e uns exames de sangue voltamos pro "+
    "resultado duas horas depois o coracao estava bem gracas a deus mas ela estava com "+
    "uma infeccao urinaria que tinha subido pro rim e precisava de antibiotico por dez "+
    "dias a medica explicou tudo direitinho e deu a receita escrita a mao que foi um "+
    "desafio decifrar na farmacia mas acabou dando certo minha mae ficou aliviada mas "+
    "tambem brava porque disse que estava com esse problema ha uma semana e eu nao "+
    "tinha percebido e ela tem razao as vezes a gente fica tao ocupada com o trabalho "+
    "e com os filhos e com mil compromissos que nao para pra olhar pras pessoas que "+
    "estao do nosso lado e que precisam de atencao voltamos pra casa as onze da noite "+
    "ela tomou o antibiotico comeu uma sopa e foi dormir eu fiquei sentada na cozinha "+
    "tomando cha e pensando em como o tempo passa rapido no ano passado ela ainda subia "+
    "escada sem dificuldade agora ja precisa segurar no corrimao essas coisas acontecem "+
    "devagar e a gente so percebe quando para pra olhar de verdade vou tentar ligar "+
    "mais e visitar mais e nao deixar que o trabalho engula tudo que e importante "+
    "hoje ela ja esta melhor tomou o remedio de manha comeu bem e ate riu de uma "+
    "bobagem que eu falei que foi bom de ver ela sorrir de novo depois de dias tao "+
    "dificeis a gente esquece o quanto um sorriso da mae pode mudar o dia inteiro",
      loadModel())
    sentences.length should be >= 2
  }

  // 2. Multilingual support: the same XLM-R backbone should also segment non-English text.
  "SentenceDetectorSaTModel" should "segment German text" taggedAs SlowTest in {
    val sentences = segment(
      "Das Leben ist wie eine Schachtel Pralinen. Man weiß nie, was man bekommt.",
      loadModel())
    sentences.length should be >= 1
  }

  // 3. Long document: more than 512 tokens forces multiple overlapping windows, which must be
  //    stitched back together without crashing and without exploding into spurious sentences.
  "SentenceDetectorSaTModel" should "handle a document longer than 512 tokens" taggedAs SlowTest in {
    val longText = (1 to 100)
      .map(i =>
        s"This is sentence number $i and it contains a few extra words to make it longer.")
      .mkString(" ")
    val sentences = segment(longText, loadModel().setBlockSize(510).setStride(256))
    sentences.length should be >= 1
  }

  // 4. Offset correctness: each annotation's begin/end must point back at its own result text.
  "SentenceDetectorSaTModel" should "produce correct begin/end character offsets" taggedAs SlowTest in {
    import spark.implicits._
    val text = "Hello world. Goodbye world."
    val data = Seq(text).toDF("text")
    val result = new Pipeline()
      .setStages(Array(documentAssembler, loadModel()))
      .fit(data)
      .transform(data)

    val annotations = result
      .selectExpr("sentence")
      .collect()
      .head
      .getAs[Seq[Row]](0)
      .map(com.johnsnowlabs.nlp.Annotation(_))

    annotations.foreach { ann =>
      text.substring(ann.begin, ann.end + 1).trim should be(ann.result.trim)
    }
  }

  // 5. Serialization round-trip: a saved model reloads with its params and still runs.
  "SentenceDetectorSaTModel" should "be saveable and reloadable" taggedAs SlowTest in {
    val savePath = "./tmp_sat_model"
    loadModel().setThreshold(0.25f).write.overwrite().save(savePath)

    val reloaded = SentenceDetectorSaTModel.load(savePath)
    reloaded.getThreshold should be(0.25f +- 1e-6f)

    segment("Hello world. Goodbye world.", reloaded).length should be >= 1
  }
}
