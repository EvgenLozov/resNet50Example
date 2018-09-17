package com.company;

import java.util.Comparator;

class IndexWeightComparator implements Comparator<IndexWeight> {
    @Override
    public int compare(IndexWeight o1, IndexWeight o2) {
        int result = Double.compare(o2.getWeight(), o1.getWeight());

        if (result == 0)
            result = Integer.compare(o1.getIndex(), o2.getIndex());

        return result;
    }
}
