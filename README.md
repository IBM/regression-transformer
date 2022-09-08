# Regression Transformer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multitask Transformer that reformulates regression as a conditional sequence modeling task.
This yields a dichotomous language model that seamlessly integrates regression with property-driven conditional generation task.

![Summary](assets/overview.jpg).

## Use the pretrained models
The Regression Transformer is implemented in the [GT4SD](https://github.com/GT4SD/gt4sd-core) library.
Via GT4SD, using several pretrained RegressionTransformers is a matter of a few lines of code :rocket 
See the tutorial [here](https://github.com/GT4SD/gt4sd-core/blob/main/notebooks/regression-transformer-demo.ipynb).
Via GT4SD you can use the RT pretrained on small molecules with some properties as shown in the paper, in particular [QED](https://www.nature.com/articles/nchem.1243) and [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x) (water solubility). There is also a multiproperty variant of the RT: a model trained jointly on logP and synthesizability (aka [SCScore](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622)).
For protein language modeling, you will also find a RT trained on a [peptide stability](https://www.science.org/doi/full/10.1126/science.aan0693) dataset from the [TAPE](https://github.com/songlab-cal/tape) benchmark.

If you use [GT4SD](https://github.com/GT4SD/gt4sd-core)) can generate molecules like this:
```
from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer, RegressionTransformerMolecules
)

buturon = "CC(C#C)N(C)C(=O)NC1=CC=C(Cl)C=C1"
target_esol = -3.53 
config = RegressionTransformerMolecules(
    algorithm_version="solubility",
    search="sample",
    temperature=2, 
    tolerance=5,
    sampling_wrapper={
        'property_goal': {'<esol>': target_esol}, 
        'fraction_to_mask': 0.2
    }
)
esol_generator = RegressionTransformer(configuration=config, target=buturon)
generations = list(esol_generator.sample(8))
```

Explore the solubility of the local chemical space around Buturon. Upon varying the property primers, you might obtain something like this:
![Esol](assets/esol.png).




## Development setup
This is mainly intended to reproduce or extend the results of the paper.
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