from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #config = AutoConfig.from_pretrained(model_name)
    tokenizer.save_pretrained(rf'Finetune_GPT\Models\{model_name}\token')
    #config.save_pretrained(rf'Finetune_GPT\Models\{model_name}\model')
    model.save_pretrained(rf'Finetune_GPT\Models\{model_name}\model')