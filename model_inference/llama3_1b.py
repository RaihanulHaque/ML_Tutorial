from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.int8,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
response = outputs[0]["generated_text"][-1]["content"]
print(response)
# Arrrr, me hearty! Yer lookin' fer a bit o' information about meself, eh? Alright then, matey! I be a language-generatin' swashbuckler, a digital buccaneer with a penchant fer spinnin' words into gold doubloons o' knowledge! Me name be... (dramatic pause)...Assistant! Aye, that be me name, and I be here to help ye navigate the seven seas o' questions and find the hidden treasure o' answers! So hoist the sails and set course fer adventure, me hearty! What be yer first question?
