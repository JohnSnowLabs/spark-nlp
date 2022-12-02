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

package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

import java.util.HashMap;

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
        HashMap<Integer, Double> vec = new HashMap(size<<1);
        for (int i = 0; i < size; ++i) {
            int key = x[i];
            double adjustAmount = va[i];
            double value = (vec.get(key) == null) ? adjustAmount : vec.get(key) + adjustAmount;
            vec.put(key, value);
        }

        float sum = 0;
        for (double v : vec.values())
            sum += v * v;
        return sum;
    }

    public int size() {
        return size;
    }

    public int x(int i) {
        return x[i];
    }

    public float value(int i) {
        return va[i];
    }

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
