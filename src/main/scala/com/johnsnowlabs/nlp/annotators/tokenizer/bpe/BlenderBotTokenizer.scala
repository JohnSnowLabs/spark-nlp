package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

class BlenderBotTokenizer(
   merges: Map[(String, String), Int],
   vocab: Map[String, Int],
   specialTokens: SpecialTokens,
   padWithSequenceTokens: Boolean = false,
   addPrefixSpaceToSentence: Boolean = false)
  extends Gpt2Tokenizer(
    merges,
    vocab,
    specialTokens,
    padWithSequenceTokens,
    prependString = "Ġ",
    addPrefixSpaceToSentence)
