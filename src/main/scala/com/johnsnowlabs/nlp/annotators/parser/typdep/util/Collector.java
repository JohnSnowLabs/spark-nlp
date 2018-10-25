package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

public interface Collector {
    void addEntry(int x);
    void addEntry(int x, float va);
}
