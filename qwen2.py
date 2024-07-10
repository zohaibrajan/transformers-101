from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cpu" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto", # what datatype comes with the model
    device_map="auto" # will know that their is no GPU and will automatically map the model to CPU
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."}, # you can elaborate
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device) # tokenize the text, returns them as PyTorch tensors, and loads them onto the device. In this case, the device is the CPU.

generated_ids = model.generate( #
    model_inputs.input_ids, # get the numerical representation of the input text
    max_new_tokens=512 # generate up to 512 tokens
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # decode the generated tokens into text
