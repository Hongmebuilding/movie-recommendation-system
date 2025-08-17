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

---
</br>

# book-recommendation-system - by Liam Song
https://www.kaggle.com/code/leesumin98/book-recommend-system-0c824f


## 1. 데이터 소개 & 준비
- **데이터**: `user_id`, `item_id(book_id)`, `rating/timestamp` + 책 메타데이터(제목, 작가, 장르/키워드, 요약)
- **전처리**: 중복/이상치 제거, 희소도 확인, 최소 인터랙션 필터링, ID 인덱싱, train/valid/test 분할
- **환경**: Python (`pandas`, `numpy`, `scipy`, `scikit-learn`, `surprise/implicit`, `PyTorch/TF`)

---

## 2. 추천시스템 개요
- **목적**: 사용자별 관련성 높은 아이템 랭킹 제공 → CTR, 유지율, 매출 개선
- **알고리즘**: 
  - Memory-based CF  
  - Matrix Factorization (MF)  
  - Deep Learning (NeuMF 등)  
  - Machine Learning 랭킹  
  - Content-based / Popularity-based
- **성능 지표**: RMSE, MAE, Precision@K, Recall@K, NDCG@K, MAP@K

---

## 3. 베이스라인 추천
- **Memory-based CF**: 사용자/아이템 간 코사인 유사도 기반 가중 평균  
  \[
  \hat r_{u,i} = \frac{\sum_{v \in N(u)} s(u,v)\, r_{v,i}}{\sum_{v \in N(u)} |s(u,v)|}
  \]
- **Matrix Factorization (SVD/NMF/SVD++)**  
  \[
  R \approx P Q^\top, \quad \hat r_{u,i} = \mu + b_u + b_i + p_u^\top q_i
  \]

---

## 4. 성능 고도화
- **Deep Learning (NCF/NeuMF)**: 사용자·아이템 임베딩 → GMF + MLP 결합  
- **Machine Learning 모델**: Gradient Boosting (GBR/XGBoost/LightGBM)으로 피처 기반 예측  

---

## 5. Cold Start 해결
- **Content-based**: 텍스트(TF-IDF/임베딩), 장르, 작가 정보 활용  
- **Popularity-based**: 전역/세그먼트 인기 점수 기반 추천  

---

## 6. 최종 모델 클래스
- **구성**: MF(SVD) + ML(GBR) + Content+Popularity 앙상블
- **인터페이스**
  ```python
  class Recommender:
      def fit(self, interactions, items): ...
      def predict(self, user, items): ...
      def recommend(self, user, k=10): ...
