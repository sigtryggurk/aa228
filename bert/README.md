# BERT

## Pre-trained models

Bert Models Used:

*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
    12-layer, 768-hidden, 12-heads, 110M parameters
    
Download the above and save it in the directory.
Each .zip file contains three items:

*   A TensorFlow checkpoint (`bert_model.ckpt`) containing the pre-trained
    weights (which is actually 3 files).
*   A vocab file (`vocab.txt`) to map WordPiece to word id.
*   A config file (`bert_config.json`) which specifies the hyperparameters of
    the model.

Before running this example you must download the
THe data we are using is [WIC data]

```shell
go the directory containing bert
export BERT_BASE_DIR=uncased_L-12_H-768_A-12
export GLUE_DIR=Data

python run_classifier.py \
  --task_name=WIC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/WIC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/wic_output/
```

This is the output:

```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.85035783
INFO:tensorflow:  eval_loss = 0.49918082
INFO:tensorflow:  global_step = 570
INFO:tensorflow:  loss = 0.4969664

```

#### Prediction from classifier

Once you have trained your classifier you can use it in inference mode by using
the --do_predict=true command. You need to have a file named test.tsv in the
input folder. Output will be created in file called test_results.tsv in the
output folder. Each line will contain output for each sample, columns are the
class probabilities.

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier

python run_classifier.py \
  --task_name=WIC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/WIC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/wic_output/
```

#### From Where?

From the following paper [the Arxiv paper](https://arxiv.org/abs/1810.04805):

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
