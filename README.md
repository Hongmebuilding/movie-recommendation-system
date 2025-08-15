# movie-recommendation-system
## 📌 핵심 기술 설명

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
문서 집합에서 특정 단어의 **중요도**를 나타내는 통계적 수치이다.  
자주 등장하지만 모든 문서에 공통적으로 나타나는 단어의 가중치는 낮추고,  
특정 문서에서만 자주 등장하는 단어의 가중치는 높힌다.

**계산 과정**
1. **TF (Term Frequency)**  
   TF(t, d) = (단어 t가 문서 d에 등장한 횟수) / (문서 d의 전체 단어 수)

2. **IDF (Inverse Document Frequency)**  
   IDF(t) = log( 전체 문서 수 N / (1 + 단어 t가 등장한 문서 수) )

3. **TF-IDF**  
   TF-IDF(t, d) = TF(t, d) × IDF(t)
---

### 2. 코사인 유사도 (Cosine Similarity)
두 벡터 간 **방향의 유사도**를 측정하는 지표이다.
문서 벡터가 이루는 각도를 이용하여, 값이 1에 가까울수록 유사도가 높음을 의미한다.

**계산식**
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

- A · B : 두 벡터의 내적
- ||A|| : 벡터 A의 크기(길이)

---

**활용**
- TF-IDF로 문서를 벡터화 → 코사인 유사도로 문서 간 유사도 계산  
- 추천 시스템, 검색 랭킹, 중복 문서 탐지 등에 활용
