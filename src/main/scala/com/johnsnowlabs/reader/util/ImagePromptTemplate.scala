package com.johnsnowlabs.reader.util

object ImagePromptTemplate {

  def getQwen2VLChatTemplate(prompt: String): String = {

    val systemMessage =
      """<|im_start|>system
         |You are a helpful assistant.<|im_end|>
         |""".stripMargin

    val userMessage =
      s"""<|im_start|>user
         |<|vision_start|><|image_pad|><|vision_end|>$prompt<|im_end|>
         |""".stripMargin

    val assistantMessage =
      """<|im_start|>assistant
         |""".stripMargin // Starts assistant response

    systemMessage + userMessage + assistantMessage
  }

  def getSmolVLMChatTemplate(prompt: String): String = {
    val userMessage =
      s"""<|im_start|>User:<image>$prompt<end_of_utterance>
         |Assistant:""".stripMargin

    userMessage
  }

  def getInternVLChatTemplate(prompt: String): String = {
    val userMessage =
      s"""<|im_start|><image>
         |$prompt<|im_end|><|im_start|>assistant
         |""".stripMargin

    userMessage
  }

  def customTemplate(template: String, prompt: String): String = {
    template.replace("{prompt}", prompt)
  }

}
