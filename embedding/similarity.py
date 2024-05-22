import numpy as np
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
import torch
from transformers import BertModel

# KoBERT 모델과 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

uploaded = pd.read_csv('q2/tax_data.csv')


# 'dockeynote' 열의 데이터를 리스트로 추출
sentences = np.load('sentences.npy')
# 저장된 임베딩 로드
all_embeddings = np.load('all_embeddings.npy')

# 코사인 유사도 계산
def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 유사 문장 찾기 함수
def find_similar_sentences(input_sentence):
    input_tokens = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        input_embeddings = model(**input_tokens)
        input_vector = input_embeddings.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

    similarities = [(i, calculate_cosine_similarity(input_vector, all_embeddings[i])) for i in range(len(all_embeddings))]
    top_20_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:20]
    return [(idx, sentences[idx], similarity) for idx, similarity in top_20_similar]

# 예제 실행
input_sentence = ""
similar_sentences = find_similar_sentences(input_sentence)
for idx, sentence, sim in similar_sentences:
    print(f"인덱스: {idx} - 유사도: {sim:.4f} - 문장: {sentence}")
