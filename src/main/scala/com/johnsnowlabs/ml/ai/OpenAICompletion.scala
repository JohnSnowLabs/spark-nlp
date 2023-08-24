/*
 * Copyright 2017-2023 John Snow Labs
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
package com.johnsnowlabs.ml.ai

import com.amazonaws.thirdparty.apache.http.client.methods.HttpPost
import com.amazonaws.thirdparty.apache.http.entity.{ContentType, StringEntity}
import com.amazonaws.thirdparty.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import com.amazonaws.thirdparty.apache.http.util.EntityUtils
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.johnsnowlabs.ml.ai.model.CompletionResponse
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader, JsonBuilder, JsonParser}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}

/** Transformer that makes a request for OpenAI Completion API for each executor.
  *
  * @see
  *   [[https://platform.openai.com/docs/api-reference/completions/create OpenAI API Doc]] for
  *   reference
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.ml.ai.OpenAICompletion
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val openAICompletion = new OpenAICompletion()
  *  .setInputCols("document")
  *  .setOutputCol("completion")
  *  .setModel("text-davinci-003")
  *  .setMaxTokens(50)
  *
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   openAICompletion
  * ))
  *
  * val promptDF = Seq(
  *  "Generate a restaurant review.",
  *  "Write a review for a local eatery.",
  *  "Create a JSON with a review of a dining experience.").toDS.toDF("text")
  * val completionDF = pipeline.fit(promptDF).transform(promptDF)
  *
  * completionDF.select("completion").show(false)
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |completion                                                                                                                                                                                                                                                                                        |
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[{document, 0, 258, \n\nI had the pleasure of dining at La Fiorita recently, and it was a truly delightful experience! The menu boasted a wonderful selection of classic Italian dishes, all exquisitely prepared and presented. The service staff was friendly and attentive and really, {}, []}]|
  * |[{document, 0, 227, \n\nI recently visited Barbecue Joe's for dinner and it was amazing! The menu had so many items to choose from including pulled pork, smoked turkey, brisket, pork ribs, and sandwiches. I opted for the pulled pork sandwich and let, {}, []}]                               |
  * |[{document, 0, 172, \n\n{ \n   "review": { \n      "overallRating": 4, \n      "reviewBody": "I enjoyed my meal at this restaurant. The food was flavourful, well-prepared and beautifully presented., {}, []}]                                                                                   |
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */

class OpenAICompletion(override val uid: String)
    extends AnnotatorModel[OpenAICompletion]
    with HasSimpleAnnotate[OpenAICompletion] {

  def this() = this(Identifiable.randomUID("OPENAI_COMPLETION"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  val model = new Param[String](this, "model", "ID of the OpenAI model to use")

  def setModel(value: String): this.type = set(model, value)

  val suffix = new Param[String](
    this,
    "suffix",
    "The suffix that comes after a completion of inserted text.")

  def setSuffix(value: String): this.type = set(suffix, value)

  val maxTokens =
    new IntParam(this, "maxTokens", "The maximum number of tokens to generate in the completion.")

  def setMaxTokens(value: Int): this.type = set(maxTokens, value)

  val temperature =
    new FloatParam(this, "temperature", "What sampling temperature to use, between 0 and 2")

  def setTemperature(value: Float): this.type = set(temperature, value)

  val topP = new FloatParam(
    this,
    "topP",
    "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass")

  def setTopP(value: Float): this.type = set(topP, value)

  val numberOfCompletions =
    new IntParam(this, "numberOfCompletions", "How many completions to generate for each prompt.")

  def setNumberOfCompletions(value: Int): this.type = set(numberOfCompletions, value)

  val logprobs = new IntParam(
    this,
    "logprobs",
    "Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.")

  def setLogprobs(value: Int): this.type = set(logprobs, value)

  val echo = new BooleanParam(this, "echo", "Echo back the prompt in addition to the completion")

  val stop = new StringArrayParam(
    this,
    "stop",
    "Up to 4 sequences where the API will stop generating further tokens.")

  def setStop(value: Array[String]): this.type = set(stop, value)

  val presencePenalty = new FloatParam(
    this,
    "presencePenalty",
    "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")

  def setPresencePenalty(value: Float): this.type = set(presencePenalty, value)

  val frequencyPenalty = new FloatParam(
    this,
    "frequencyPenalty",
    "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")

  def setFrequencyPenalty(value: Float): this.type = set(frequencyPenalty, value)

  val bestOf = new IntParam(
    this,
    "bestOf",
    "Generates best_of completions server-side and returns the `best` (the one with the highest log probability per token).")

  def setBestOf(value: Int): this.type = set(bestOf, value)

  val logitBias: StructFeature[Option[Map[String, Int]]] =
    new StructFeature[Option[Map[String, Int]]](this, "logitBias")

  def setLogitBias(value: Option[Map[String, Int]]): this.type = set(logitBias, value)

  val user = new Param[String](
    this,
    "user",
    "A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.")

  def setUser(value: String): this.type = set(user, value)

  private var bearerToken: Option[Broadcast[String]] = None

  def setBearerTokenIfNotSet(spark: SparkSession, openAIKey: Option[String]): this.type = {
    if (bearerToken.isEmpty && openAIKey.isDefined) {
      bearerToken = Some(spark.sparkContext.broadcast(openAIKey.get))
    }
    this
  }

  def getBearerToken: String = {
    if (bearerToken.isDefined) bearerToken.get.value else ""
  }

  setDefault(
    maxTokens -> 16,
    temperature -> 1f,
    topP -> 1f,
    numberOfCompletions -> 1,
    echo -> false,
    presencePenalty -> 0f,
    frequencyPenalty -> 0f,
    bestOf -> 1)

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    this.setBearerTokenIfNotSet(
      dataset.sparkSession,
      Some(ConfigLoader.getConfigStringValue(ConfigHelper.openAIAPIKey)))
    dataset
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val prompts = annotations.map(annotation => annotation.result)
    val logitBiasString = getLogitBiasAsJsonString

    val suffixJson = JsonBuilder.formatOptionalField("suffix", get(suffix))
    val maxTokensJson = JsonBuilder.formatOptionalField("max_tokens", get(maxTokens))
    val temperatureJson = JsonBuilder.formatOptionalField("temperature", get(temperature))
    val topPJson = JsonBuilder.formatOptionalField("top_p", get(topP))
    val numberOfCompletionsJson = JsonBuilder.formatOptionalField("n", get(numberOfCompletions))
    val logprobsJson = JsonBuilder.formatOptionalField("logprobs", get(logprobs))
    val echoJson = JsonBuilder.formatOptionalField("echo", get(echo))
    val stopJson = JsonBuilder.formatOptionalField("stop", get(stop))
    val presencePenaltyJson =
      JsonBuilder.formatOptionalField("presence_penalty", get(presencePenalty))
    val frequencyPenaltyJson =
      JsonBuilder.formatOptionalField("frequency_penalty", get(frequencyPenalty))
    val bestOfJson = JsonBuilder.formatOptionalField("best_of", get(bestOf))
    val logitBiasJson = JsonBuilder.formatOptionalField("logit_bias", logitBiasString)
    val userJson = JsonBuilder.formatOptionalField("user", get(user))

    val jsonTemplate =
      """
        |{
        |    "model": "%s",
        |    "prompt": "%s",
        |    "stream": false
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |    %s
        |}
        |""".stripMargin

    val jsons = prompts.map(prompt =>
      jsonTemplate.format(
        $(model),
        prompt,
        suffixJson,
        maxTokensJson,
        temperatureJson,
        topPJson,
        numberOfCompletionsJson,
        logprobsJson,
        echoJson,
        stopJson,
        presencePenaltyJson,
        frequencyPenaltyJson,
        bestOfJson,
        logitBiasJson,
        userJson))
    val openAIUrlCompletion = "https://api.openai.com/v1/completions"
    val annotationsCompletion = jsons.map { json =>
      val response = post(openAIUrlCompletion, json)
      Annotation(DOCUMENT, 0, response.length, response, Map())
    }
    annotationsCompletion
  }

  private def getLogitBiasAsJsonString: Option[String] = {
    if (get(logitBias).isDefined) {
      val objectMapper = new ObjectMapper().registerModule(DefaultScalaModule)
      Some(objectMapper.writeValueAsString($$(logitBias)))
    } else None
  }

  private def post(url: String, jsonBody: String): String = {
    val httpPost = new HttpPost(url)
    httpPost.setEntity(new StringEntity(jsonBody, ContentType.APPLICATION_JSON))
    val bearerToken = getBearerToken
    require(bearerToken.nonEmpty, "OpenAI API Key required")
    httpPost.setHeader("Authorization", s"Bearer $bearerToken")

    var text: String = null
    var responseBody: String = ""

    val httpclient: CloseableHttpClient = HttpClients.createDefault()
    try {
      val response = httpclient.execute(httpPost)
      responseBody = EntityUtils.toString(response.getEntity)
      val completionResponse = JsonParser.parseObject[CompletionResponse](responseBody)
      text = completionResponse.choices.head.text
    } catch {
      case ex: Exception =>
        if (responseBody.contains("error"))
          throw new Exception(responseBody)
        else ex.printStackTrace()
    } finally {
      httpclient.close()
    }

    text
  }

}
