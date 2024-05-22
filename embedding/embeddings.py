import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel
from kobert_tokenizer import KoBERTTokenizer

# CSV 파일 읽기 (ZIP 파일 처리는 주석 처리함)
uploaded = pd.read_csv('tax_data.csv')

# 'dockeynote' 열의 데이터를 리스트로 추출
sentences = uploaded['dockeynote'].fillna('').astype(str).tolist()

# GPU 사용 가능 여부 확인 및 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# KoBERT 모델과 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_states=True)
model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=True)
model.to(device)  # 모델을 GPU로 이동

# 임베딩 계산
embeddings = []
batch_size = 1024
for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        embeddings.append(batch_embeddings)

    torch.cuda.empty_cache()

# 임베딩을 하나의 큰 배열로 결합
all_embeddings = np.vstack([emb.numpy() for emb in embeddings])

# 임베딩을 npy 파일로 저장
np.save('all_embeddings.npy', all_embeddings)
print('all_embddings.npy saved.')
np.save('sentences.npy', np.array(sentences))
print('sentences.npy saved.')
