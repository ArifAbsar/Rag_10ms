from langchain.chat_models import init_chat_model
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM
import torch
#pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",trust_remote_code=True,)
text_gen = pipeline(
    "text-generation",
    model="bigscience/bloomz-560m",
    device=0,
    do_sample=True,
    temperature=0.3,
)

# 2) LangChain wrapper
llm = HuggingFacePipeline(
    pipeline=text_gen,
    model_kwargs={
        "max_new_tokens": 100,
    }
)
