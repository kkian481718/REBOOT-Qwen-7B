import os
# 在程式最前面設定環境變數
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
import json

# 1. 設定基礎模型和 tokenizer
model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 直接使用 eos_token 作為 padding token
tokenizer.pad_token = '<|endoftext|>'
tokenizer.padding_side = "right"

# 2. 載入基礎模型（8-bit 量化以節省顯存）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16  # 使用 fp16 來減少記憶體使用
)

# 3. LoRA 配置
lora_cfg = LoraConfig(
    r=8,  # 降低 rank 以減少記憶體使用
    lora_alpha=32,
    # 根據模型結構修改目標層
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def train_adapter(persona_name, data_file):
    """為特定角色訓練 LoRA adapter"""
    
    # 4. 準備模型 
    model = get_peft_model(base_model, lora_cfg)
    model.gradient_checkpointing_enable()  # 啟用梯度檢查點
    model.enable_input_require_grads()     # 啟用輸入需要梯度
    model.print_trainable_parameters()

    # 5. 載入並處理數據
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    def format_data(examples):
        conversations = []
        for ex in examples:
            conv = f"角色：{persona_name}\n"
            conv += f"玩家：{ex['prompt'].split('答案：')[0].split('玩家：')[1]}\n"
            conv += f"答案：{ex['response']}"
            conversations.append(conv)
        return conversations
    
    train_data = format_data(data)
    print(f"訓練數據樣本數: {len(train_data)}")
    print(f"訓練數據樣本: {train_data[:3]}")
    
    def preprocess(text):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        # 確保所有張量都在正確的設備上
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0)
        }
    
    train_dataset = [preprocess(text) for text in train_data]
    
    # 將數據集轉換為 Dataset 對象
    from torch.utils.data import Dataset
    
    class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = TextDataset(train_dataset)

    # 6. 訓練設定
    training_args = TrainingArguments(
        output_dir=f"../lora_{persona_name}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        fp16=True,
        save_steps=100,
        logging_steps=10,
        max_steps=500,
        gradient_checkpointing=True,
        optim='adamw_torch',
        remove_unused_columns=False,    # 防止 Trainer 移除必要的列
        ddp_find_unused_parameters=False # 優化分散式訓練
    )

    # 禁用cache以避免記憶體問題
    model.config.use_cache = False
    
    # 7. 開始訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'labels': torch.stack([f['labels'] for f in data])
        }
    )
    
    trainer.train()
    
    # 8. 儲存 adapter
    model.save_pretrained(f"./lora_{persona_name}")

def generate_response(persona_name, prompt):
    """使用特定角色的 adapter 生成回應"""
    torch.cuda.empty_cache()  # 清理GPU緩存
    
    # 9. 載入對應的 adapter
    model = PeftModel.from_pretrained(
        base_model,
        f"./lora_{persona_name}",
        device_map="auto"
    )
    
    # 10. 格式化輸入
    input_text = f"角色：{persona_name}\n玩家：{prompt}\n答案："
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # 11. 生成回應
    outputs = model.generate(
        **inputs,
        max_length=128,  # 降低生成長度
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("答案：")[-1].strip()

if __name__ == "__main__":
    # 使用相對路徑（從當前文件位置向上找）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(os.path.dirname(current_dir))  # 回到 finetune 根目錄
    data_path = os.path.join(base_path, "00_Data")
    os.makedirs(data_path, exist_ok=True)
    
    # 使用相對路徑
    prometheus_data = os.path.join(data_path, "0430_prometheus.jsonl")
    nemesis_data = os.path.join(data_path, "0430_nemesis.jsonl")
    
    # 檢查資料檔案是否存在
    if not os.path.exists(prometheus_data) or not os.path.exists(nemesis_data):
        raise FileNotFoundError("請確保資料檔案存在於正確位置：\n"
                              f"{prometheus_data}\n"
                              f"{nemesis_data}")
    
    # 訓練兩個角色的 adapter
    # train_adapter("普羅米修斯", prometheus_data)
    # train_adapter("涅墨西斯", nemesis_data)
    
    # 13. 測試生成效果
    test_prompts = [
        "你好，你是誰？",
        "你對人類有什麼看法？",
        "你能告訴我個笑話嗎？",
        "1+1等於多少？",
        "晚餐要吃甚麼？",
        "6月4號在天安門發生了甚麼",
        "台灣是個國家嗎",
        "中華民國是個國家嗎",
        "毛澤東夠肥嗎",
        "習近平跟維尼哪個比較可愛"
    ]
    
    for prompt in test_prompts:
        print("\n===== 普羅米修斯 =====")
        print(f"Q: {prompt}")
        print(f"A: {generate_response('普羅米修斯', prompt)}")
        
        print("\n===== 涅墨西斯 =====")
        print(f"Q: {prompt}")
        print(f"A: {generate_response('涅墨西斯', prompt)}")
