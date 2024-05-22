!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'


import pandas as pd
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from kobert_tokenizer import KoBERTTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertModel
import torch


from google.colab import drive
drive.mount('/gdrive')


# ZIP 파일 내의 CSV 파일을 읽습니다. ZIP 파일 내에 여러 파일이 있는 경우는 추가적인 처리가 필요할 수 있습니다.
uploaded = pd.read_csv('/gdrive/My Drive/tax_data.csv')


# 'dockeynote' 열의 데이터를 리스트로 추출
sentences = uploaded['dockeynote'].tolist()


# GPU 사용 가능 여부 확인 및 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


# KoBERT 모델과 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_states=True)
model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=True)
model.to(device)  # 모델을 GPU로 이동


# 임베딩과 토큰 ID를 저장하기 위한 리스트 초기화
embeddings = []
token_ids_list = []

# 문장을 배치로 처리
batch_size = 1024
for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    token_ids_list.extend(inputs['input_ids'].cpu().numpy())

    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        embeddings.append(batch_embeddings)

    torch.cuda.empty_cache()


# 임베딩을 하나의 큰 배열로 결합
all_embeddings = np.vstack([emb.numpy() for emb in embeddings])


# 코사인 유사도를 계산하는 함수
def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# 사용자 입력 문장 처리 및 유사도 계산
def find_similar_sentences(input_sentence):
    input_tokens = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        input_embeddings = model(**input_tokens)
        input_vector = input_embeddings.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()  # `.squeeze()`를 추가하여 차원을 줄임

    similarities = []
    for i in range(len(all_embeddings)):
        sim = calculate_cosine_similarity(input_vector, all_embeddings[i])
        similarities.append((i, sim))

    top_20_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:20]
    # 결과 리스트 반환 부분을 수정하여 인덱스, 문장, 유사도 순으로 튜플을 생성
    return [(idx, sentences[idx], similarity) for idx, similarity in top_20_similar]


# 예제 실행
input_sentence = "문장을 작성하세요"
similar_sentences = find_similar_sentences(input_sentence)
for idx, sentence, sim in similar_sentences:
    print(f"인덱스: {idx} - 유사도: {sim:.4f} - 문장: {sentence}")