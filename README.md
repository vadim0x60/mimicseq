# Intensive Care as One Big Sequence Modeling Problem

## Understand

MIMIC-Ext-SEQ is a benchmark for foundation models in Intensive Care, representing the journey of an intensive care patient as a sequence of event tokens with optional event intensity markers designed to make it easy to train sequence models (Transformers, etc.). See [the paper](https://vadim.me/publications/mimicseq) for more details.

## Get access

- [Obtain access to MIMIC IV dataset](https://console.cloud.google.com/storage/browser/mimicseq) if you haven’t already. You will have to sign up on PhysioNet, accept the data use agreement and take a small course.
- Go to your profile settings on PhysioNet, select the “Cloud” tab and specify your Google account.
- [Download the data from here](https://console.cloud.google.com/storage/browser/mimicseq).

## Evaluate your model

See `example.py`

## Re-create our dataset

`reproduce` directory contains the Google BigQuery SQL requests that we ran on MIMIC IV database to create our dataset. It is not required to use MIMIC-SEQ, but included for transparency and reproducibility.
