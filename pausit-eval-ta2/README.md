Pausit Eval TA2

<p align="center">
  <img src="assets/pausit.svg" width="600" height="300"/>
</p>

This repository contains the combined code of the official T&E evaluation with our TA2 systems

# Setup

Clone the repo:

```bash
git clone git@github.com:iarpa-hiatus/pausit-eval-ta2.git
```

Create an environment by running:
```bash
python3.11 -m venv venv/
source venv/bin/activate
```
Which will create a directory called `venv/` which will store all the dependencies. Now run:
```bash
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
```
Which will install all the dependencies for the project.

You won't have access to the internet during evaluation. So, download and store the weights of models to use it. (You have the option to just do it for the model you want to run)

For baseline:
```bash
mkdir sentence-transformers
cd sentence-transformers
git lfs install
git clone https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1
cd ..

mkdir rrivera1849
cd rrivera1849
git clone https://huggingface.co/rrivera1849/LUAR-MUD
cd ..
```

For SRS:
```bash
mkdir longformer
cd longformer
git lfs install
git clone https://huggingface.co/allenai/longformer-base-4096
cd ..
mv -r longformer author_attribution/
```
You should be good to go!

# Usage
## Data

The HRS data is too large to host on Github, so you can find it here: [https://drive.google.com/drive/u/0/folders/1tr9rtfZIA5zQfJLo_4SFdMC44IwILmG9](https://drive.google.com/drive/u/0/folders/1tr9rtfZIA5zQfJLo_4SFdMC44IwILmG9)

## Evaluation pipeline


This system is designed to execute multiple scripts from one script `main.py`:
```                                      
usage: main.py [-h] [--helpfull] --input-dir INPUT_DIR --output-dir OUTPUT_DIR --ground-truth-dir GROUND_TRUTH_DIR --run-id RUN_ID
               [--query-identifier QUERY_IDENTIFIER] [--candidate-identifier CANDIDATE_IDENTIFIER] [--debug]
               [-ta1 {lex2vec,datadreamer_lora,baseline_sbert,baseline_luar,none}] [-g] -ta2 {srs,baseline}

options:
  -h, --help              show this help message and exit
  --input_dir             the directory containing pre-generated sample query and candidate json files
  --output-dir            Directory where the query and candidate attributions will be stored
  --ground-truth-dir      Directory where the ground truth query and candidate attributions are stored
  --run-id                Run identifier
  --query-identifier      Identifier for query embeddings. If running at author level, set to authorIDs
  --candidate-identifier  Identifier for candidate embeddings. If running at author level, set to authorSetIDs
  -ta1,                   --ta1-approach{baseline_sbert,none}
                          Which TA1 approach to evaluate
  -g                      Option to genearate TA1 features using the TA1 approach mentioned
  -ta2                    --ta2-approach{baseline,srs}
                          Which TA2 approach to evaluate
```


At the end, the results and config from both eval systems will be printed to terminal, for which you can copy and paste into here: TBA.

## Working example

TBA