from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import psutil
from abc import ABC
import warnings
import os
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig

class load_model(ABC):
    def __init__(self):
  
        self.original_model = None
        self.dataset = None
        self.tokenizer  = None
        self.dialogue = None
        
    def load_dataset_and_llm(self):
        huggingface_dataset_name = "knkarthick/dialogsum"
        dataset = load_dataset(huggingface_dataset_name)

        model_name='google/flan-t5-base'

        original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map= "auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.original_model  = original_model
        self.dataset  = dataset
        self.tokenizer  = tokenizer

        return self.original_model, self.tokenizer, self.dataset 

    def print_number_of_trainable_model_parameters(self, model):
        #NOTE model = self.original_model
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")
        
        del trainable_model_params
        del all_model_params
        del param 
        return f"ram temizlendi"    

    def test_the_model_with_zero_shot_inferencing(self, original_model, tokenizer, dataset):
        """
        Modeli sıfır atış çıkarımıyla test eder.

        Bu fonksiyon, belirtilen modelin sıfır atış çıkarımını gerçekleştirir ve sonuçları analiz eder. Modelin temel özetle karşılaştırıldığında
        diyaloğu özetlemekte zorlandığını görebilirsiniz, ancak metinden, modelin eldeki göreve göre ince ayar yapılabileceğini gösteren bazı
        önemli bilgileri çıkardığını belirtir.
        """
        index = 200

        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary:
        """

        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
            original_model.generate(
                inputs["input_ids"].cuda(original_model.device.index), 
                max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
        )

        dash_line = '-'.join('' for x in range(100))
        print(dash_line)
        print(f'INPUT PROMPT:\n{prompt}')
        print(dash_line)
        print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
        print(dash_line)
        print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

        return dialogue 

    def tokenize_function(self, example):
        index = 200
        dialogue = self.dataset['test'][index]['dialogue']
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example['input_ids'] = self.tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda(0)
        example['labels'] = self.tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda(0)
        
        return example
    
    def setup_tokenize(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
        tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

        return tokenized_datasets

    def check_dataset(self,tokenized_datasets): 
        print(f"Shapes of the datasets:")
        print(f"Training: {tokenized_datasets['train'].shape}")
        print(f"Validation: {tokenized_datasets['validation'].shape}")
        print(f"Test: {tokenized_datasets['test'].shape}")

        print(tokenized_datasets)

    def finetune_model(self,original_model, tokenized_datasets):

        output_dir = f'GenerativeAI/dialogue-summary-training-{str(int(time.time()))}'

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=1,
            max_steps=1
        )

        trainer = Trainer(
            model=original_model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation']
        )

        # trainer.train()
        """Training a fully fine-tuned version of the model would take a few hours
           on a GPU. To save time, download a checkpoint of the fully fine-tuned
           model to use in the rest of this notebook. This fully fine-tuned model
           will also be referred to as the **instruct model** 
        """
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained("GenerativeAI/flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16, device_map={"":0})
        return instruct_model

class peft:
    def __init__(self, instance:load_model):
   
        self.original_model, self.tokenizer, self.dataset = instance.load_dataset_and_llm()
        self.tokenized_datasets = instance.setup_tokenize(self.dataset)

    def setup_peft(self):
        lora_config = LoraConfig(
            r=32, # Rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )

        peft_model = get_peft_model(self.original_model, 
                                    lora_config)
        
        print(self.print_number_of_trainable_model_parameters(peft_model))

        return peft_model

    def train_peft_adapter(self, peft_model):
        
        # output_dir = f'GenerativeAI/peft-dialogue-summary-training-{str(int(time.time()))}'
        output_dir = f'C:/Users/sbnkr/Desktop/Generative_AI/GenerativeAIforNLP/dialogue-summary-training-{str(int(time.time()))}'

        peft_training_args = TrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3, # Higher learning rate than full fine-tuning.
            num_train_epochs=1,
            logging_steps=1,
            max_steps=1    
        )
        

        peft_trainer = Trainer(
            model= peft_model,
            args=peft_training_args,
            train_dataset=self.tokenized_datasets["train"],
        )

        peft_trainer.train()

        peft_model_path="C:/Users/sbnkr/Desktop/Generative_AI/GenerativeAIforNLP/peft-dialogue-summary-checkpoint-local"

        peft_trainer.model.save_pretrained(peft_model_path)
        self.tokenizer.save_pretrained(peft_model_path)


    def model_load(self):

        peft_model_path="C:/Users/sbnkr/Desktop/Generative_AI/GenerativeAIforNLP/peft-dialogue-summary-checkpoint-local"
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16, device_map= "auto")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        peft_model = PeftModel.from_pretrained(peft_model_base, 
                                            peft_model_path, 
                                            torch_dtype=torch.bfloat16,
                                            is_trainable=False)
        print(self.print_number_of_trainable_model_parameters(peft_model))
        return peft_model

    def print_number_of_trainable_model_parameters(self, model):
        #NOTE model = self.original_model
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")
        
        del trainable_model_params
        del all_model_params
        del param 
        return f"ram temizlendi"   
    
    def test(self, peft_model, original_model, dataset , tokenizer):

        index = 200
        dialogue = dataset['test'][index]['dialogue']
        # baseline_human_summary = dataset['test'][index]['summary']
        human_baseline_summary = dataset['test'][index]['summary']

        prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary: """

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda(0)

        original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

        peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

        dash_line = '-'.join('' for x in range(100))
        print(dash_line)
        print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
        print(dash_line)

        print(f'ORIGINAL MODEL:\n{original_model_text_output}')

        print(dash_line)
        print(f'PEFT MODEL: {peft_model_text_output}')

        print()

    def test1(self, peft_model, original_model, dataset , tokenizer):
        dialogues = dataset['test'][0:10]['dialogue']
        human_baseline_summaries = dataset['test'][0:10]['summary']

        original_model_summaries = []
        peft_model_summaries = []

        for idx, dialogue in enumerate(dialogues):
            prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary: """

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda(0)

  

            original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

            peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
            peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

            original_model_summaries.append(original_model_text_output)

            peft_model_summaries.append(peft_model_text_output)

        zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))

        df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
        print(df)

class control_mechanism:
    def __init__(self):
        pass

    def cuda_gpu_control(self):

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f"GPU adı: {torch.cuda.get_device_name(device)}")
        else:
            print("GPU kullanılamıyor.")

        os.system("nvidia-smi")

    def ram_control(self):
        # Sistem bellek kullanımını al
        memory_info = psutil.virtual_memory()

        # Toplam RAM miktarı
        total_memory = memory_info.total

        # Kullanılan RAM miktarı
        used_memory = memory_info.used

        # Boş RAM miktarı
        available_memory = memory_info.available

        print(f"Toplam RAM: {total_memory / (1024 ** 3):.2f} GB")
        print(f"Kullanılan RAM: {used_memory / (1024 ** 3):.2f} GB")
        print(f"Boş RAM: {available_memory / (1024 ** 3):.2f} GB")


if __name__ == "__main__":

    original_model_args = load_model()
    original_model, tokenizer, dataset = original_model_args.load_dataset_and_llm()
 

    peft_model_args = peft(original_model_args)
    # peft_model=peft_model_args.setup_peft()
    # peft_model_args.train_peft_adapter(peft_model)


    peft_model = peft_model_args.model_load()
    peft_model_args.test(peft_model, original_model, dataset , tokenizer)
    peft_model_args.test1(peft_model, original_model, dataset , tokenizer)
