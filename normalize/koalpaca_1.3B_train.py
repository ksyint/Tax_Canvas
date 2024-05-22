import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import glob 
import json 
from datasets import Dataset,DatasetDict
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model



files=glob.glob("normalized/*/*.json")
main_list=[]
# 모든 json 파일들을 가져온다. 
for index,file in enumerate(files):
  
    dict={}
    with open(file,"r") as label:
        label2=json.load(label)
        
    dict["instruction"]=label2["docKeynote"]
    dict["output"]=label2["docType_lawType"]
    
    main_list.append(dict)
  
# instruction 이 input 이고 output이 ground truth 이다. 
# 이 때 instruction 은 keynote이며 해당 keynote의 lawtype을 alpaca가 예측을 하는것이다. 


data=Dataset.from_list(main_list)

data = data.map(
    lambda x: {'text': f"### 질문: 다음 문장의 클래스를 분류하시오. ### {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" }
)

model_id = "EleutherAI/polyglot-ko-1.3b"  # safetensors 컨버팅된 레포

# 12b  이 가장 성능이 좋아서 쓰려고 하다가 메모리 에러
# 따라서 이를 더 경량화시도를 하려고 했지만 시간이 없어서 1b 으로 바꿔서 train 시작 
# 현재 데이터 전처리는 더 손댈것이 없음 
# checkpoint 다 나오면 inference 시도도 해보겠습니다. 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 4비트로 로드하는지 여부를 나타내는 매개변수
    bnb_4bit_use_double_quant=True, # 이중 양자화를 사용하는지 여부
    bnb_4bit_quant_type="nf4", # 4비트 양자화의 유형을 지정하는 매개변수
    bnb_4bit_compute_dtype=torch.bfloat16 # 4비트 양자화에 사용되는 연산의 데이터 타입
)

tokenizer = AutoTokenizer.from_pretrained(model_id) #tokenizer를 load한다. 
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
# 주어진 모델 ID에 해당하는 양자화 구성을 적용한 모델을 로드
# 앞서 생성한 양자화 설정을 모델에 적용


data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)
# 데이터셋 내의 모든 텍스트를 tokenizer에 거쳐서 수치화된 토큰으로 전환한다. 


#model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
# 모델을 k-비트 학습에 준비합니다. 이는 모델을 k-비트 양자화 및 학습을 수행할 수 있도록 준비


config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# LoraConfig 클래스를 사용하여 모델의 설정을 구성합니다.
# r은 LORA (Learnable Quantization Range)의 값을 설정합니다.
# lora_alpha는 LORA의 알파 값을 설정합니다. ==> 알파값은 일종의 스케일링 인자로 작용하여 가중치 값의 범위를 조절합니다. 작은 알파값은 더 작은 양자화 범위를 의미하며, 이는 더 많은 양자화 수준으로 인해 더 많은 정보가 손실될 수 있음을 의미합니다. 반면에 큰 알파값은 더 큰 양자화 범위를 의미하며, 더 적은 양자화 수준으로 인해 더 적은 정보가 손실될 수 있습니다.
# 따라서 알파값은 양자화의 세분화 수준을 제어하여 모델의 정확성과 성능 사이의 균형을 조절하는 데 사용됩니다.

# target_modules은 양자화를 적용할 모듈을 지정합니다.
# lora_dropout은 LORA의 드롭아웃 값을 설정합니다.
# bias는 모델의 편향을 설정합니다.
# task_type은 작업 유형을 설정합니다. 여기서는 CAUSAL_LM (인과 언어 모델)으로 설정되어 있습니다.

model = get_peft_model(model, config)


tokenizer.pad_token = tokenizer.eos_token
# 토크나이저의 패딩 토큰을 문장 종료 토큰(eos_token)으로 설정합니다. eos : end of sentence 
# 이는 패딩 토큰을 문장의 끝으로 설정하여 모델 학습 및 추론에 사용될 수 있도록 합니다

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=20,
        gradient_accumulation_steps=4, # 그래디언트 누적 스텝을 설정합니다. 이는 몇 번의 배치 처리 후에 그래디언트를 업데이트할 지를 결정합니다. 
                                       # 여기서는 4로 설정되어 있으므로, 4번의 배치 처리 후에 그래디언트를 업데이트합니다
        max_steps=500000, 
        learning_rate=1e-4,
        fp16=True,

      # FP16 (반정밀도) 학습을 사용할지 여부를 결정합니다. 
      # FP16 학습은 메모리 사용량을 줄이고 학습 속도를 향상시키는데 도움이 됩니다. 
      # FP16 학습을 사용하려면 GPU가 반정밀도를 지원해야 합니다.
      
        logging_steps=20,
        save_steps=520,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # 데이터 수집기로, 토크나이저와 언어 모델링을 위한 데이터 수집 방법을 지정
)

model.config.use_cache = False 
# silence the warnings. Please re-enable for inference!
# 모델의 캐시 사용 여부를 False로 설정합니다. 이는 경고를 제거하기 위한 것이며, 
# 추론시에는 다시 활성화할 것을 권장합니다

trainer.train(resume_from_checkpoint=True)

