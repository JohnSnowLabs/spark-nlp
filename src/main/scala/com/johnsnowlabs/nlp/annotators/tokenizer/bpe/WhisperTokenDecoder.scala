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

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import java.nio.charset.Charset

/** Class used by Whisper model to decode tokens. Does not require merges and is therefore
  * omitted.
  *
  * Note that this means this class cannot tokenize strings.
  *
  * @param vocab
  *   Vocabulary of Tokens for decoding
  * @param specialTokens
  *   Special tokens that should be filtered during decoding
  */
class WhisperTokenDecoder(vocab: Map[String, Int], specialTokens: SpecialTokens)
    extends Gpt2Tokenizer(Map.empty, vocab, specialTokens) {

  /** Decodes the token ids into strings.
    *
    * Token IDs, that are not in the vocabulary are decoded to an empty string (some token IDs
    * might not be in the vocabulary).
    *
    * @param tokens
    *   Array of token IDs to decode
    * @return
    *   Decoded string
    */
  override def decodeTokens(tokens: Array[Int]): String = {
    val text = tokens
      .map(token => decoderVocab.getOrElse(token, ""))
      .filter(x => !specialTokens.contains(x))
      .mkString("")
    val bytes = text.map(x => unicodeToByteMapping(x.toString)).map(x => x.toByte).toArray
    new String(bytes, Charset.forName("UTF-8"))
  }
}
