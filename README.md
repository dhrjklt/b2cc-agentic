# B2CC Agent in LangGraph + Open WebUI

This agent is configured to work with local inference engine vLLM but can be integrate with other cloud based inference API's.

## Set Up
1. Create environment using uv
2. Install dependencies with uv
```shell
uv pip install requirements.txt
```
3. Set up vLLM Server:
```shell
vllm  serve meta-llama/Llama-3.1-8B-Instruct   --dtype auto    --max-model-len 32000   --max-num-seqs 8   --max-num-batched-tokens 8192  --enable-auto-tool-choice --tool-call-parser llama3_json
```
4. Set up Gitlab and MySQL connection string in `.env` file (which should be located in `tools/`). 
```shell
PDB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=
PROJECT_ID=
GITLAB_PRIVATE_TOKEN=
```

## Run
1. Activate your uv environment
2. Simply run:
```shell
source .venv/bin/activate
sh start.sh
```
3. Open Open WebUI and make sure your pipeline is connected to it.
4. Use 'B2CC Agent (Streaming)' agent as a model to chat.

