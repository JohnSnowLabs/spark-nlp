package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

// TODO: How to do this properly?
private[nlp] class BpeSpecialTokens(modelType: String) {
  val availableModels = Array("roberta")
  require(availableModels.contains(modelType), "Model type \"" + modelType + "\" not supported yet.")
  def getSentencePadding: (String, String) =
    modelType match {
      case "roberta" => ("<s>", "</s>")
      //      "gpt2" -> Map("cls_token_id" -> "<|endoftext|>", "sep_token_id" -> "<|endoftext|>")
    }
  def getSpecialTokens: Map[String, TokenTransformations] =
    modelType match {
      case "roberta" => SpecialTokens.robertaSpecialTokens
    }
}

private object SpecialTokens {
  val robertaSpecialTokens: Map[String, TokenTransformations] = Map(
    "<s>" -> TokenTransformations(
      normalized = true,
      id = 0,
      singleWord = false,
      rstrip = false,
      content = "<s>",
      lstrip = false
    ),
    "<pad>" -> TokenTransformations(
      normalized = true,
      id = 1,
      singleWord = false,
      rstrip = false,
      content = "<pad>",
      lstrip = false
    ),
    "</s>" -> TokenTransformations(
      normalized = true,
      id = 2,
      singleWord = false,
      rstrip = false,
      content = "</s>",
      lstrip = false
    ),
    "<unk>" -> TokenTransformations(
      normalized = true,
      id = 3,
      singleWord = false,
      rstrip = false,
      content = "<unk>",
      lstrip = false
    ),
    "<mask>" -> TokenTransformations(
      normalized = true,
      id = 50264,
      singleWord = false,
      rstrip = false,
      content = "<mask>",
      lstrip = true
    )
  )
}

case class TokenTransformations(
    normalized: Boolean,
    id: Int,
    singleWord: Boolean,
    rstrip: Boolean,
    lstrip: Boolean,
    content: String
) {}
