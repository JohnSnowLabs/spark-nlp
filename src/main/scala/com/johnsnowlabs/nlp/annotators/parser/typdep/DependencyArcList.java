package com.johnsnowlabs.nlp.annotators.parser.typdep;

public class DependencyArcList {
    private int n;
    private int[] st;
    private int[] edges;

    DependencyArcList(int[] heads)
    {
        n = heads.length;
        st = new int[n];
        edges = new int[n];
        constructDepTreeArcList(heads);
    }

    int startIndex(int i)
    {
        return st[i];
    }

    int endIndex(int i)
    {
        return (i >= n-1) ? n-1 : st[i+1];
    }

    public int get(int i)
    {
        return edges[i];
    }

    private void constructDepTreeArcList(int[] heads)
    {

        for (int i = 0; i < n; ++i)
            st[i] = 0;

        for (int i = 1; i < n; ++i) {
            int j = heads[i];
            ++st[j];
        }

        for (int i = 1; i < n; ++i)
            st[i] += st[i-1];

        for (int i = n-1; i > 0; --i) {
            int j = heads[i];
            --st[j];
            edges[st[j]] = i;
        }
    }
}
