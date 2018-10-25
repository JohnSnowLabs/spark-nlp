package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

public class ScoreCollector implements Collector {

    private float[] weights;
    private float score;

    public float getScore() {
        return score;
    }

    public ScoreCollector(float[] w) {
        weights = w;
        score = 0;
    }

    @Override
    public void addEntry(int x) {
        score += weights[x];
    }

    @Override
    public void addEntry(int x, float va) {
        score += weights[x]*va;
    }

}
