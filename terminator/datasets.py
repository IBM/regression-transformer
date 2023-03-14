from transformers import LineByLineTextDataset, PreTrainedTokenizer, TextDataset


def get_dataset(
    filepath: str,
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    line_by_line: bool = True,
):
    if line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=filepath, block_size=block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=filepath,
            block_size=block_size,
        )
