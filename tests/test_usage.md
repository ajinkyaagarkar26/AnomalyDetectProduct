# How to Use the Updated deeplog.py

The `deeplog.py` script has been updated to support passing vocab size from the vocab mode to the predict mode. Here's how to use it:

## 1. Generate vocabulary and get vocab size

```bash
python3 deeplog.py vocab
```

This will:
- Process the vocabulary from the training data
- Save the vocabulary to `vocab.pkl`
- Print the vocab size
- Exit with the vocab size as the exit code

## 2. Use the vocab size in predict mode

```bash
python3 deeplog.py predict --vocab_size 24
```

This will:
- Use the specified vocab size (24 in this example) instead of the default
- Update the model configuration with the new vocab size
- Run the prediction with the correct vocabulary size

## 3. You can also use mean and std parameters

```bash
python3 deeplog.py predict --vocab_size 24 --mean 0.5 --std 1.2
```

## 4. Example workflow to capture vocab size and use it

```bash
# Step 1: Run vocab mode and capture the vocab size
python3 deeplog.py vocab
VOCAB_SIZE=$?  # This captures the exit code which is the vocab size

# Step 2: Use the vocab size in predict mode
python3 deeplog.py predict --vocab_size $VOCAB_SIZE
```

## Changes Made

1. **Added `--vocab_size` parameter** to the predict parser
2. **Modified predict mode** to accept and use the vocab_size parameter
3. **Updated `process_vocab` function** to return the vocab size
4. **Added parameter handling** for mean and std in predict mode
5. **Made vocab mode exit** with the vocab size as the exit code

The script will automatically recreate the model with the correct vocab size when you provide the `--vocab_size` parameter to the predict mode.
