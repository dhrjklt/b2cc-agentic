# B2CC Agent in LangGraph + Open WebUI

This agent is set up to use the local vLLM inference engine by default, but it can also be integrated with other cloud-based inference APIs as needed.

## Set Up
1. Create environment using uv
2. Install dependencies with uv
```shell
uv pip install requirements.txt
```
3. Set up vLLM Server:
```shell
vllm  serve Qwen/Qwen2.5-7B-Instruct  --dtype auto --enable-auto-tool-choice --tool-call-parser llama3_json
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

