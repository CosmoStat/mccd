#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mccd.dataset_generation as dataset_generation
from joblib import Parallel, delayed, cpu_count, parallel_backend


# Total number of datasets (equivalent to number of proceses)
n_procs = 4
# Number of CPUs to use
n_cpus = 2

# Print some info
cpu_info = ' - Number of available CPUs: {}'.format(cpu_count())
proc_info = ' - Total number of processes: {}'.format(n_procs)

# Paths
e1_path = './e1_psf.npy'
e2_path = './e2_psf.npy'
fwhm_path = './seeing_distribution.npy'
output_path = './datasets/'

# Print infof
print('Dataset generation.')
print(cpu_info)
print(proc_info)
print('Number of catalogs: ', n_procs)
print('Number of CPUs: ', n_cpus)

# Generate catalog list
cat_id_list = [2000000 + i for i in range(n_procs)]


def generate_dataset(cat_id):
    print('\nProcessing catalog: ', cat_id)
    sim_dataset_generator = dataset_generation.GenerateRealisticDataset(
        e1_path=e1_path,
        e2_path=e2_path,
        size_path=fwhm_path,
        output_path=output_path,
        catalog_id=cat_id)
    sim_dataset_generator.generate_train_data()
    sim_dataset_generator.generate_test_data()


with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpus)(delayed(generate_dataset)(_cat_id)
                                        for _cat_id in cat_id_list)
