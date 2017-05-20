
import tensorflow as tf
import numpy as np
import tempfile

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex

# Write all examples into a TFRecords file
with tempfile.NamedTemporaryFile() as fp:
    writer = tf.python_io.TFRecordWriter(fp.name)
    for sequence, label_sequence in zip(sequences, label_sequences):
        ex = make_example(sequence, label_sequence)
        writer.write(ex.SerializeToString())
    writer.close()
# A single serialized example
# (You can read this from a file using TFRecordReader)
ex = make_example([1, 2, 3], [0, 1, 0]).SerializeToString()

# Define how to parse the example
context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

# Parse the example
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
)
# [0, 1, 2, 3, 4 ,...]
x = tf.range(1, 10, name="x")

# A queue that outputs 0,1,2,3,...
range_q = tf.train.range_input_producer(limit=5, shuffle=False)
slice_end = range_q.dequeue()

# Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
y = tf.slice(x, [0], [slice_end], name="y")

# Batch the variable length tensor with dynamic padding
batched_data = tf.train.batch(
    tensors=[y],
    batch_size=5,
    dynamic_pad=True,
    name="y_batch"
)

# Run the graph
# tf.contrib.learn takes care of starting the queues for us
res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)

# Print the result
print("Batch shape: {}".format(res[0]["y"].shape))
print(res[0]["y"])
