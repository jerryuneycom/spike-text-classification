# spike-text-classification
+ Use Pytorch to build and predict the output.
+ For model version 2. Use the xlm-roberta-base pretrained model as the tokenizer

## Install environment
### Install python3
Follow python official document https://www.python.org/ or install by brew https://docs.brew.sh/Homebrew-and-Python

### Install dependencies
```
$ pip install -r requirements.txt
```

## Config
Look at the config file to see the configuration. Change the model version to switch between versions.

## Run
### Build model
```
$ python build.py
```

### Evaluate model
```
$ python evaluate.py
```

## Note
### Can't run the model in the mps device
Change the device code line to the cpu
```
# device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = torch.device("cpu")
```