package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

import com.johnsnowlabs.nlp.annotators.parser.Collector;
import gnu.trove.map.hash.TIntDoubleHashMap;

public class FeatureVector implements Collector {

    private int size = 0;

    private int capacity;
    private int[] x;
    private float[] va;

    public FeatureVector() {
        initCapacity();
    }

    private void initCapacity() {
        this.capacity = 40;
        x = new int[40];
        va = new float[40];
    }

    private void grow() {

        int cap = 5 > capacity ? 10 : capacity * 2;

        int[] x2 = new int[cap];
        float[] va2 = new float[cap];

        if (capacity > 0) {
            System.arraycopy(x, 0, x2, 0, capacity);
            System.arraycopy(va, 0, va2, 0, capacity);
        }

        x = x2;
        va = va2;
        capacity = cap;
    }

    @Override
    public void addEntry(int x) {
        if (size == capacity) grow();
        this.x[size] = x;
        va[size] = 1.0f;
        ++size;
    }

    @Override
    public void addEntry(int x, float value) {
        if (value == 0) return;

        if (size == capacity) grow();
        this.x[size] = x;
        this.va[size] = value;
        ++size;
    }

    public void addEntries(FeatureVector m) {
        addEntries(m, 1.0f);
    }

    public void addEntries(FeatureVector m, float coeff) {

        if (coeff == 0 || m.size == 0) return;

        for (int i = 0; i < m.size; ++i)
            addEntry(m.x[i], m.va[i] * coeff);
    }

    public float squaredL2NormUnsafe()
    {
        TIntDoubleHashMap vec = new TIntDoubleHashMap(size<<1);
        for (int i = 0; i < size; ++i)
            vec.adjustOrPutValue(x[i], va[i], va[i]);
        float sum = 0;
        for (double v : vec.values())
            sum += v*v;
        return sum;
    }

    public int size() {
        return size;
    }

    public int x(int i) { return x[i]; }
    public float value(int i) { return va[i]; }
    public float dotProduct(float[] y) {
        return dotProduct(this, y);
    }

    private static float dotProduct(FeatureVector featureVector, float[] y) {

        float sum = 0;
        for (int i = 0; i < featureVector.size; ++i)
            sum += featureVector.va[i] * y[featureVector.x[i]];
        return sum;
    }

}
