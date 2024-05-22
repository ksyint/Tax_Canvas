from similarity import find_similar_sentences


similar_sentences = find_similar_sentences("쟁점금액을 중개수수료로 보아 양도차익 계산 시 필요경비로 인정할 수 있는지 여부")
for idx, sentence, sim in similar_sentences:
    print(f"인덱스: {idx} - 유사도: {sim:.4f} - 문장: {sentence}")
