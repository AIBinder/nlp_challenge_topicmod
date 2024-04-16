# SUMM AI NLP Challenge - Topic Modeling with LLM 🚀

## Initial setup
1. Download fine-tuned model
   * apt-get install git-lfs; git lfs install
   * mkdir models/; cd models/
   * git clone https://huggingface.co/AI-Binder/topic_gen_v1
2. Launch Docker Environment (docker compose up -d) [after cd .. to project root folder]


Optional for fine-tuning:
3. Install and activate venv (python -m venv venv; source venv/bin/activate)
4. Install Packages (pip install -r requirements.txt)
5. Install flash-attention for higher performance (pip install flash-attn==2.5.7 --no-build-isolation)

## Dataset
- MLSum (https://huggingface.co/datasets/mlsum)
- German Texts with annotated topics and summaries 
- (including texts with topic 'München' :D)

## Hardware Requirements for Fine-Tuning
- GPU with at least 24 GB VRAM (e.g., A10)
- ideally Ampere Architecture to support bfloat16 


# task description

## Objective
Fine-tune an open-source Large Language Model (LLM) such as Llama2 to perform topic modeling on a collection of text paragraphs. The model should generate a meaningful topic for a given text.

## Dataset
There is no dataset provided. Find a relevant dataset related to this task.

## Submission
Please clone this repository and upload it to a new private repo.
Implement a well-organized codebase along with a README documenting the setup, key findings, and challenges.  
Add me (@flowni) to the repo for submitting it.
You have one week to complete the assignment. ⏰

## Evaluation Criteria
The focus of this assignment is on the development process and the ability to fine-tune an LLM. Don't worry too much about the performance of the model as long as you know how to improve it.
The following criteria will be used to evaluate the assignment:

1. **Code Quality:**
   - Readability and organization of the code. 📦

2. **Documentation:**
   - Clarity and completeness of the documentation. 📚

3. **(Bonus for Docker environment) 🐳**

## Contact
Feel free to reach out to me (Nicholas) for any questions related to the assignment. 📧

Have fun! 😊
