import joblib
from bertopic import BERTopic
from torch import bfloat16
import transformers

itr = 2
min_cluster_size = 100
file_path = "/mnt/scratch/lellaom/narrative_theme_identification/results/itr"+str(itr)+"_"+str(min_cluster_size)+"/"

model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_compute_dtype=bfloat16  
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto'
)
model.eval()
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""
example_prompt = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure to only return the label and nothing more.

[/INST]
"""
prompt = system_prompt + example_prompt + main_prompt
topic_info = joblib.load(file_path+"topic_info_df.joblib")
llama_topic_label = []
for idx, row in topic_info.iterrows():
    documents = ''
    for doc in row['Representative_Docs']:
        doc = '- '+doc+'.\n'
        documents += doc
    keywords = ', '.join(row['Representation'])
    new_prompt = prompt
    new_prompt = new_prompt.replace('[DOCUMENTS]', documents)
    new_prompt = new_prompt.replace('[KEYWORDS]', keywords)
    label = generator(new_prompt)[0]["generated_text"].split("[/INST]")[-1]
    label = label.strip()
    llama_topic_label.append(label)

topic_info.insert(2, "Narrative", llama_topic_label)
doc_info = joblib.load(file_path+"doc_info_df.joblib")
min_topic = topic_info["Topic"].min()
if min_topic == 0:
    doc_labels = [llama_topic_label[topic] for topic in doc_info["Topic"].values]
elif min_topic == -1:
    doc_labels = [llama_topic_label[topic+1] for topic in doc_info["Topic"].values]
doc_info.insert(3, "Narrative", doc_labels)
topic_info.to_csv(file_path+"topic_info.csv", index=False)
topic_info.to_excel(file_path+"topic_info.xlsx", index=False)
doc_info.to_csv(file_path+"doc_info.csv", index=False)
doc_info.to_excel(file_path+"doc_info.xlsx", index=False)