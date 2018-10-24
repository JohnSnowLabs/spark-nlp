package com.johnsnowlabs.nlp.annotators.parser;

public interface Collector {
    void addEntry(int x);
    void addEntry(int x, float va);
}
