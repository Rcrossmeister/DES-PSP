# DES-PSP
 Double Encoder Seq2seq-Based Presidential Conceptual Stock Price Prediction model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README Still need a further revise and implements.* 

*Note: Reproduce paper list is need to be determined.*

*Note: The test parameter need to be appointed.*

__Update:__

* *Refactored the whole repo. 10-25-2023*
* *A argparse interface is released. 09-015-2023*

**TO-DO:**

1. Ablation Study.
2. Need to compare the proposed DES-PSP model with existing state-of-the-art methods in the field of stock market prediction.
3. Need more comprehensive analysis of the model's sensitivity to different parameter settings.
4. Potentially combine novelty methods (After well-done baseline).
5. Verifying the hypothesis of the exists stock which benefits from the election of a specific presidential candidate.
6. Find background news to support the election of conceptual stocks.
7. State the generalizability.

## DataSet

The full dataset : ./data/raw/raw/All_Data.csv

## Use

* To **train** the Seq2Seq baseline model :

```shell
python main_baseline.py --model=seq2seq_lstm
```


## Citation

Please cite our paper if you use it in your work:

```shell
@inproceedings{,
   title={{}: },
   author={},
   booktitle={},
   year={}
}
```
