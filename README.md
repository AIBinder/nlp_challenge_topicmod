# News Topic Classification via Fine-tuning Mistral-7B LLM ☀️

## Initial Setup
1. Download the fine-tuned LoRA adapter for the Mistral-7B model into llm_inference/models/ folder
   * apt-get install git-lfs; git lfs install
   * cd llm_inference/; mkdir models/; cd models/
   * git clone https://huggingface.co/AI-Binder/topic_gen_v1   
   
   [Optional] 1b. If you want to use the official Mistral-Repo for the base model, a file '.env' has to be created in the llm_inference-Folder with content HF_READ_TOKEN="hf_..." (using a valid Huggingface Access-Token of an account that has accepted the access conditions at http://huggingface.co/mistralai/Mistral-7B-v0.1)  
2. Launch Docker Environment (docker compose up -d)  
[in project root folder]

Then the app should be accessible at http://localhost:8501.

(Docker version 24.0.7 and Docker Compose version 2.21.0 used, should allow GPU access for docker container as defined in the docker-compose file)

[Optional] Additional steps for fine-tuning:  
3. Install and activate venv (python -m venv venv; source venv/bin/activate)  
4. Install Packages (pip install -r requirements.txt)  
5. Install flash-attention (pip install flash-attn==2.5.7 --no-build-isolation)

Then the fine-tuning can be started via 'python llm_finetune.py'

(Python 3.10 used for virtual environment)

## Short Project Description
- Mistral-7B Base-Model is fine-tuned on Topics + Titles of German news articles (cf. annotated Dataset below)
- The fine-tuned Model is provided in a docker environment with LLM inference and a simple streamlit frontend

## Dataset
- MLSum (https://huggingface.co/datasets/mlsum)
- German Texts with annotated topics, titles and summaries 

## Hardware Requirements for Fine-Tuning
- GPU with at least 24 GB VRAM (e.g., NVIDIA A10)
- (CUDA Version 12.2 used for Fine-tuning)

## Key findings/ challenges
- Desired structure of the Topic-Description in German to be determined more thoroughly for more precise fine-tuning (e.g., "Das Thema ist " + Artikel + substantivierter Begriff + Erläuterung in einem kurzen Folgesatz)
- For that, the dataset needs to be further preprocessed or a accordingly annotated dataset created by labeling (if a suited dataset is not publicly available) 

## Next steps
- Improve the fine-tuning with more training data and structured experiments
- Control the output of the LLM-Inference with, e.g., stop_sequences, frequency penalties and manual checks
- Integrate the training data and the texts entered in the frontend into VectorDB to be able to search texts with similar topics
