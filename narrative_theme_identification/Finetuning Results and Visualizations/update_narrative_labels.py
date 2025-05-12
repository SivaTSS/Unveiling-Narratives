import pandas as pd
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from tqdm import tqdm

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=bfloat16)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "/mnt/scratch/lellaom/models"
tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                          cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             cache_dir=cache_dir,
                                             trust_remote_code=True,
                                             quantization_config=bnb_config,
                                             device_map='auto',)
model.eval()
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
generator = pipeline(tokenizer=tokenizer,
                     model=model,
                     task="text-generation",
                     do_sample=False,
                     max_new_tokens=50,)

chat = [
    {"role": "system", "content": "You are a helpful assistant. Your task is to identify the main theme of the narrative."}
]
example_dict = {"role": "user"}
example_content = """I have a narrative that contains the following documents:
- Climate change is causing extreme weather patterns worldwide, leading to flooding, droughts, and wildfires.
- Global warming continues to raise temperatures, affecting agriculture and ecosystems, leading to food insecurity.
- The rise in carbon emissions is linked to increased deforestation, reducing the planet's ability to absorb CO2.

The narrative is characterized by the following keywords: ['climate', 'change', 'warming', 'emissions', 'temperature', 'deforestation', 'extreme', 'weather', 'flooding', 'ecosystems'].

Based on the information about the narrative, please provide the main theme of the narrative in a few words. Make sure to return only the main theme of the narrative and nothing more.
"""
example_dict['content'] = example_content
chat.append(example_dict)
reply_dict = {"role": "assistant", "content": "Impact of climate change on the environment and society"}
chat.append(reply_dict)
user_content = """I have a narrative that contains the following documents:
[DOCUMENTS]
The narrative is characterized by the following keywords: [KEYWORDS].

Based on the information about the narrative, please provide the main theme of the narrative in a few words. Make sure to return only the main theme of the narrative and nothing more.
"""
df = pd.read_csv("secret_adversary_doc_info.csv")
cluster_theme = {}
for topic in tqdm(set(df['Topic']), desc="Inference Running"):
    info_dict = {"role": "user"}
    content = user_content
    docs = list(df[df['Topic']==topic]['Document'])
    docs = docs[0:10]
    documents = ''
    for doc in docs:
        doc = '- '+doc+'.\n'
        documents += doc
    content = content.replace("[DOCUMENTS]", documents)
    keywords = df[df['Topic']==topic].iloc[0]['Representation']
    content = content.replace("[KEYWORDS]", keywords)
    info_dict["content"] = content
    new_chat = chat.copy()
    new_chat.append(info_dict)
    theme = generator(new_chat)[0]["generated_text"][-1]["content"]
    cluster_theme[topic] = theme

theme = []
df = pd.read_csv("secret_adversary_doc_info.csv")
df = df.rename(columns={'Narrative': 'Narrative Theme'})
for idx, row in df.iterrows():
    theme.append(cluster_theme[row["Topic"]])
df["Narrative Theme"] = theme

df.to_csv("secret_adversary_doc_info_new.csv", index=False)