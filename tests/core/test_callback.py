import numpy as np

import peal


def test_bestworst():
    tracker = peal.callback.BestWorst()

    indiv1 = peal.Individual(np.array([]))
    indiv1.fitness = 1
    indiv2 = peal.Individual(np.array([]))
    indiv2.fitness = 2
    indiv3 = peal.Individual(np.array([]))
    indiv3.fitness = 3
    indiv4 = peal.Individual(np.array([]))
    indiv4.fitness = 4
    indiv5 = peal.Individual(np.array([]))
    indiv5.fitness = 5
    indiv6 = peal.Individual(np.array([]))
    indiv6.fitness = 6
    indiv7 = peal.Individual(np.array([]))
    indiv7.fitness = 7

    tracker.on_start(peal.Population())
    tracker.on_generation_end(peal.Population([indiv1, indiv3, indiv6]))
    tracker.on_generation_end(peal.Population([indiv5, indiv1]))
    tracker.on_generation_end(peal.Population([indiv4, indiv7, indiv2]))

    assert tracker.best.size == 3

    assert tracker.best[0] is indiv6
    assert tracker.best[1] is indiv5
    assert tracker.best[2] is indiv7

    assert tracker.worst[0] is indiv1
    assert tracker.worst[1] is indiv1
    assert tracker.worst[2] is indiv2
