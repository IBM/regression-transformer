# Regression Transformer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Codebase to experiment with a hybrid Transformer that combines conditional sequence generation with regression

![Summary](assets/overview.jpg).


## Development setup
```console
conda env create -f conda.yml
conda activate terminator
pip install -e .
```

## Generate some data
Example data for QED can be generated using [scripts/generate_example_data.py](./scripts/generate_example_data.py).
```console
python scripts/generate_example_data.py examples/example.smi examples/qed_property_example.txt
```

If you need to create a new vocabulary for a dataset you can use [scripts/create_vocabulary.py](./scripts/create_vocabulary.py) it will also automatically add some special tokens at the top of your vocabulary file.
```console
python scripts/create_vocabulary.py examples/qed_property_example.txt examples/vocab.txt
```

At this point the folder containing the vocabulary file can be used to load a tokenizer compatible with any `ExpressionBertTokenizer`:
```python
>>> from terminator.tokenization import ExpressionBertTokenizer
>>> tokenizer = ExpressionBertTokenizer.from_pretrained('examples')
>>> text = '<qed>0.3936|CBr'
>>> tokens = tokenizer.tokenize(text)
>>> print(tokens)
['<qed>', '_0_0_', '_._', '_3_-1_', '_9_-2_', '_3_-3_', '_6_-4_', '|', 'C', 'Br']
>>> token_indexes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
>>> print(token_indexes)
[16, 17, 18, 28, 45, 34, 35, 19, 15, 63]
>>> tokenizer.build_inputs_with_special_tokens(token_indexes)
[12, 16, 17, 18, 28, 45, 34, 35, 19, 15, 63, 13]
```

Prepare some train/eval data line by line:
```console
head -n 900 examples/qed_property_example.txt > examples/train.txt
tail -n +901 examples/qed_property_example.txt > examples/eval.txt
```

Launch the training:
```console
python scripts/run_language_modeling.py --output_dir examples/models/xlnet_selfies \
    --config_name configs/xlnet_selfies.json --tokenizer_name ./examples/vocab.txt \
    --do_train --do_eval --learning_rate 1e-4 --num_train_epochs 5 --save_total_limit 2 \
    --save_steps 500 --per_gpu_train_batch_size 16 --evaluate_during_training --eval_data_file ./examples/eval.txt \
    --train_data_file ./examples/train.txt --line_by_line --block_size 510 --seed 42 --logging_steps 250
```

Exemplary model configurations (number of heads, layers, etc.) can be found in the [configs](./configs) folder.


## Citation
If you use the regression transformer, please cite:
```bib
@article{born2022regression,
  title={Regression Transformer: Concurrent Conditional Generation and Regression by Blending Numerical and Textual Tokens},
  author={Born, Jannis and Manica, Matteo},
  journal={arXiv preprint arXiv:2202.01338},
  note={Spotlight talk at ICLR workshop on Machine Learning for Drug Discovery},
  year={2022}
}
```