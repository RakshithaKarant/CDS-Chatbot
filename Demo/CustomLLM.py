import os
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomerSupportTransformersLLM(LLM):
    def __init__(self, mistral_models_path):
        # Initialize Mistral _tokenizer
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(os.path.join(mistral_models_path, "token"))
        self.model = AutoModelForCausalLM.from_pretrained(os.path.join(mistral_models_path, "model") )


    def _call(self, prompt: str, stop=None):
        # Create a ChatCompletionRequest with the user message
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

        # Encode the prompt into tokens
        tokens = self._tokenizer.encode_chat_completion(completion_request).tokens

        # Generate response tokens
        out_tokens, _ = generate(
            [tokens],
            self.model,
            max_tokens=1000,  # adjust token limit as needed
            temperature=0.5,
            eos_id=self._tokenizer.instruct_tokenizer._tokenizer.eos_id
        )

        # Decode generated tokens into text
        result = self._tokenizer.decode(out_tokens[0])
        return result

    @property
    def _llm_type(self):
        return "customer_support"
