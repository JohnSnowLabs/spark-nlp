package com.johnsnowlabs.nlp.sentencepiece;

/**
 * @author https://github.com/levyfan
 * @see https://github.com/levyfan/sentencepiece-jni
 */
public class SentencePieceException extends RuntimeException {

    public SentencePieceException() {
        super();
    }

    public SentencePieceException(String message) {
        super(message);
    }

    public SentencePieceException(String message, Throwable cause) {
        super(message, cause);
    }
}
