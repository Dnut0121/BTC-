# BTC

BTC의 과거 가격을 이용한 딥러닝 모델 공부

## 목차 (Table of Contents)
1. [프로젝트 개요](#프로젝트-개요)
2. [Git 커밋 컨벤션 (Conventional Commits)](#git-커밋-컨벤션-conventional-commits)
3. [LSTM & GRU 비교](#lstm--gru-비교)
   - [LSTM](#lstm)
   - [GRU](#gru)
4. [피처 엔지니어링 (Feature Engineering)](#피처-엔지니어링-feature-engineering)

---

## 프로젝트 개요
이 프로젝트는 **딥러닝 모델**을 사용하여 특정 작업을 수행하기 위한 저장소입니다.  
학습 모델로 **GRU**를 선택하였으며, 피처 엔지니어링 과정을 통해 데이터 전처리와 변환을 진행합니다.

---

## Git 커밋 컨벤션 (Conventional Commits)
이 프로젝트는 아래와 같은 **커밋 메시지** 규칙을 따릅니다.

- **feat**: 새로운 기능 추가  
- **fix**: 버그 수정  
- **docs**: 문서 수정  
- **style**: 코드 포맷, 세미콜론 누락 등 스타일 수정  
- **refactor**: 코드 리팩토링  
- **test**: 테스트 관련 코드 추가/수정  
- **chore**: 빌드 업무, 패키지 매니저 설정 등


## LSTM & GRU 비교

LSTM은 **input gate**, **forget gate**, **output gate** 총 세 가지 게이트를 갖고 있으며, 셀 상태(cell state)와 은닉 상태(hidden state)를 분리하여 장기 의존성을 좀 더 풍부하게 학습할 수 있습니다.

GRU는 **update gate**, **reset gate** 두 가지 게이트만 사용하고, 셀 상태를 별도로 두지 않고 은닉 상태(hidden state)만을 유지합니다. 구조가 상대적으로 간단합니다.

학습에 사용할 컴퓨팅 자원(GPU)이 제한적이고, 개발/실험 시간이 많지 않기 때문에 GRU를 선택했습니다.

---

## 피처 엔지니어링 (Feature Engineering)

**정의**:  
원시 데이터를 모델이 잘 학습할 수 있도록 가공하고 변환하는 과정입니다.  
즉, 데이터에서 의미 있는 정보를 추출하거나, 불필요한 정보를 제거하여 모델에 입력되는 피처(특성)를 개선합니다.

**예시**:
- **정규화/표준화**: 데이터를 일정한 범위나 분포로 변환하여 학습 효율을 높임.
- **파생 피처 생성**: 기존 데이터로부터 새로운 피처(예: 이동 평균, 변화율 등)를 만들어 모델에 추가.
- **범주형 변수 인코딩**: 문자나 범주형 데이터를 숫자형 데이터로 변환 (예: One-hot encoding).

**목적**:  
피처 엔지니어링을 통해 모델이 데이터의 패턴을 더 잘 학습할 수 있도록 도와주고, 결과적으로 모델의 성능을 향상시킵니다.


