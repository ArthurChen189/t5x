# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ranking tasks for T5X."""

from typing import Mapping

import seqio
import t5
import tensorflow as tf
import rax
from functools import partial
import numpy as np

def _msmarco_preprocessor(dataset: tf.data.Dataset,
                          output_features: Mapping[str, t5.data.Feature],
                          shuffle_lists: bool = True) -> tf.data.Dataset:
  """Preprocessor for MS-Marco listwise ranking task.

  Args:
    dataset: The ms_marco/v2.1 dataset.
    output_features: A mapping for each of the output features.
    shuffle_lists: If True, lists are shuffled which enforces uniformly random
      selection of documents when lists are truncated to a fixed list size.

  Returns:
    The processed dataset.
  """

  def extract_inputs(features):
    # Extract features from dataset and convert to model inputs.
    query = features["query"]
    passages = features["passages"]["passage_text"]
    inputs = tf.strings.join(["Query:", query, "Document:", passages],
                             separator=" ")
    label = tf.cast(features["passages"]["is_selected"], dtype=tf.float32)
    #qid = tf.cast(features["query_id"], dtype=tf.int32)
    #pid = tf.cast(features["passages"]["passage_id"], dtype=tf.int32)
    qid = features["query_id"]
    pid = features["passages"]["passage_id"]
    print(f"pid dtype is {type(pid)}")
    # The target is an unused token to obtain a ranking score.
    targets = tf.fill(tf.shape(label), "<extra_id_10>")

    # Shuffle lists, this enforces uniformly random selection of documents when
    # lists are later truncated to a fixed list size.
    if shuffle_lists:
      shuffle_idx = tf.random.shuffle(tf.range(tf.shape(label)[0]), seed=73)
      label = tf.gather(label, shuffle_idx)
      inputs = tf.gather(inputs, shuffle_idx)
      targets = tf.gather(targets, shuffle_idx)
      pid = tf.gather(pid, shuffle_idx)
      #qid = tf.gather(qid, shuffle_idx)

    output_features = {
        "inputs": inputs,
        "targets": targets,
        "label": label,
        "mask": tf.ones_like(label, dtype=tf.bool),
        "qid": qid,
        "pid": pid
    }
    print("GOO")
    return output_features

  # Extract necessary inputs from MS-Marco dataset.
  dataset = dataset.map(extract_inputs, num_parallel_calls=tf.data.AUTOTUNE)

  # Tokenize only the text features, leave the others unchanged.
  tokenize_features = {
      "inputs": output_features["inputs"], "targets": output_features["targets"]
  }
  dataset = seqio.preprocessors.tokenize(dataset, tokenize_features)
  return dataset


def mrr_at_n(targets, scores, topk=10):
    # Compute MRR@N
    print(targets)
    print(scores)
    print(topk)
    _, indices = tf.math.top_k(scores, k=topk)
    print(indices)
    print(tf.gather(targets, indices))
    return tf.reduce_mean(tf.math.reciprocal(tf.cast(tf.gather(targets, indices), tf.float32)))


_OUTPUT_FEATURES = {
    "inputs":
        t5.data.Feature(
            vocabulary=t5.data.get_default_vocabulary(),
            add_eos=False,
            required=False,
            rank=2),
    "targets":
        t5.data.Feature(
            vocabulary=t5.data.get_default_vocabulary(),
            add_eos=False,
            rank=2),
    "label":
        t5.data.Feature(
            vocabulary=seqio.PassThroughVocabulary(size=0),
            add_eos=False,
            required=False,
            dtype=tf.float32,
            rank=1),
    "mask":
        t5.data.Feature(
            vocabulary=seqio.PassThroughVocabulary(size=0),
            add_eos=False,
            required=False,
            dtype=tf.bool,
            rank=1)
}


t5.data.TaskRegistry.add(
    "msmarco_qna21_ranking",
    seqio.Task,
    source=seqio.TfdsDataSource(
        tfds_name="huggingface:ms_marco/v2.1", splits=["train", "validation"]),
    preprocessors=[
        _msmarco_preprocessor
    ],
    output_features=_OUTPUT_FEATURES)


t5.data.TaskRegistry.add(
    "msmarco_v1_passage_ranking",
    seqio.Task,
    source=seqio.TfdsDataSource(
        tfds_name="msmarco_datasets:1.0.0", splits=["dev", "dev_top10", "test", "test_top10", "train"]),
    preprocessors=[
        _msmarco_preprocessor
    ],
    output_features=_OUTPUT_FEATURES)


t5.data.TaskRegistry.add(
    "msmarco_v1_passage_ranking_eval_only",
    seqio.Task,
    source=seqio.TfdsDataSource(
        tfds_name="msmarco_datasets:1.0.0",
        splits=["dev", "dev_top10", "test", "test_top10"]),
    preprocessors=[
        _msmarco_preprocessor
    ],
    output_features=_OUTPUT_FEATURES,
    metric_fns=[partial(mrr_at_n, topk=10)])

t5.data.TaskRegistry.add(
    "msmarco_v1_passage_ranking_test",
    seqio.Task,
    source=seqio.TfdsDataSource(
        tfds_name="only50:1.0.0", splits=["dev", "test", "train"]),
    preprocessors=[
        _msmarco_preprocessor
    ],
    output_features=_OUTPUT_FEATURES)
