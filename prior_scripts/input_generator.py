#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:44:06 2020

@author: bariskuru
"""
import numpy as np
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
import matplotlib.pyplot as plt


def inhom_poiss(mod_rate=10, max_rate=100, n_inputs=400, dur=0.5):
    """Generate an inhomogeneous poisson spike train with a rate profile that
    is a sine wave whose rate is given by the rate parameter and that maximum
    frequency is given by the max_rate parameter in Hz.
    min frequency is always 0Hz
    """

    sampling_interval = 0.0001 * pq.s

    t = np.arange(0, dur, sampling_interval.magnitude)

    rate_profile = (np.sin(t*mod_rate*np.pi*2-np.pi/2) + 1) * max_rate / 2

    rate_profile_as_asig = AnalogSignal(rate_profile,
                                        units=1*pq.Hz,
                                        t_start=0*pq.s,
                                        t_stop=dur*pq.s,
                                        sampling_period=sampling_interval)

    spike_trains = []
    for x in range(n_inputs):
        curr_train = stg.inhomogeneous_poisson_process(rate_profile_as_asig)
        # We have to make sure that there is sufficient space between spikes.
        # If there is not, we move the next spike by 0.1ms
        spike_trains.append(curr_train)

    array_like = np.array([np.around(np.array(x.times)*1000, decimals=1) for x in spike_trains])
    for arr_idx in range(array_like.shape[0]):
        bad_idc = np.argwhere(np.diff(array_like[arr_idx]) == 0).flatten()
        bad_idc = bad_idc+1
        while bad_idc.any():
            for bad_idx in bad_idc:
                array_like[arr_idx][bad_idx] = array_like[arr_idx][bad_idx] + 0.1
            bad_idc = np.argwhere(np.diff(array_like[arr_idx]) == 0).flatten()
            bad_idc = bad_idc + 1

    return array_like

array_like = inhom_poiss()

