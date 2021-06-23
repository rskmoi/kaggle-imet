# kaggle-imet
## about
Code for 23rd place solution in [Kaggle iMet Collection 2019](https://www.kaggle.com/c/imet-2019-fgvc6).

**NOTICE**

(1) Reproducibility

I have refactored this code so that it can be used in other competitions.

So reproducibility cannot be guaranteed.

(2) Model addition

Swin Transformer Model was added in 2021 for verification purposes and was not included in the competition solution.

## usage
### directory structure
```
input/
    train/
        xxx.png
    test/
        yyy.png
    train.csv
result/
     result_will_be_in_here
this repository/
    …
```

### preprocess
```
python preprocess.py
```

### training
```
python train.py --config_path=xxx.yml
```

### validation
```
python validate.py --config_path=xxx.yml
```

### submit
```
python submit.py inference --config_path=xxx.yml
python submit.py submit --dir_pickles=foo --threshold=0.XX 
```
