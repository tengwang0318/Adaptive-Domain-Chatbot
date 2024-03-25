import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


def get_model(config: argparse.PARSER):
    model = config.model_name
    print('\nDownloading model: ', model, '\n\n')

    if model == 'wizardlm':
        model_repo = 'TheBloke/wizardLM-7B-HF'

        tokenizer = AutoTokenizer.from_pretrained(model_repo)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config=bnb_config,
            device_map='auto',
            low_cpu_mem_usage=True
        )

        max_len = 1024

    elif model == 'llama2-7b-chat':
        model_repo = 'daryl149/llama-2-7b-chat-hf'

        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config=bnb_config,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        max_len = 2048

    elif model == 'llama2-13b-chat':
        model_repo = 'daryl149/llama-2-13b-chat-hf'

        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config=bnb_config,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        max_len = 2048  # 8192

    elif model == 'mistral-7B':
        model_repo = 'mistralai/Mistral-7B-v0.1'

        tokenizer = AutoTokenizer.from_pretrained(model_repo)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config=bnb_config,
            device_map='auto',
            low_cpu_mem_usage=True,
        )

        max_len = 1024
    elif model == "phi-2":
        model_repo = 'microsoft/phi-2'
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config=bnb_config,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        max_len = 2048

    else:
        raise ValueError("Not implemented model (tokenizer and backbone)")

    return tokenizer, model, max_len


def get_pipeline(model, tokenizer, max_len, config: argparse.PARSER):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_length=max_len,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
