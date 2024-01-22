import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import portpicker
import tensorflow as tf
import time
from madb.vgg16_cifar10 import model_builder, dataset_fn
from threading import Thread
from DatasetIterator import DatasetIterator


N_WORKER = 8
N_STEPS = 5

def main():

  threads = []
  mean_step_times = []

  dataset = dataset_fn()
  di = DatasetIterator(dataset, 32)

  for i in range(N_WORKER):
    model, _, forward_pass, build_optimizer, _, _, bs, lr = model_builder()
    optimizer = build_optimizer(lr)
    mean_step_times.append(0)

    my_batches = []
    for _ in range(N_STEPS):
      my_batches.append(next(di))

    def work(idx):
      sum_step_time = 0
      for j in range(N_STEPS):
        start_time = time.perf_counter()
        gradients, loss = forward_pass(my_batches[j])
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        end_time = time.perf_counter()
        sum_step_time += end_time - start_time

      mean_step_time = sum_step_time / N_STEPS
      mean_step_times[idx] = mean_step_time
  
    t = Thread(target=work, args=(i,))
    threads.append(t)

  for t in threads:
    t.start()

  for t in threads:
    t.join()

  print(mean_step_times)


if __name__ == '__main__':
  main()