# MCGAT: Metapath-based Cross-type Syncrhonized Graph Attention Network

# MCGAT
We developed a Metapath-based Cross-type Synchronized Graph Attention Network, MCGAT for herb-disease association prediction.
We applied novel 'Cross-type synchronization' and 'Incremental metapath optimization' in MCGAT.

## Requirements
Python version
* `python` == 3.11.5


## Required packages
* `pandas` == 2.0.3
* `numpy` == 1.24.3
* `torch` == 2.1.2+cu121
* `dgl` == 2.0.0+cu121


## Required input files
Input files need to run the codes. These files should be in the `data` folder.

* `coconut_he_cp.csv` - The relationships between herb and compound from COCONUT

* `coconut_he_ph.csv` - The relationships between herb and phenotype from COCONUT

* `cp_cp_id.csv` - The relationships between compound and compound from CODA

* `cp_ph_id.csv` - The relationships between compound and phenotype from CODA

* `ph_ph_id.csv` - The relationships between phenotype and phenotype from CODA

## Run analysis
Run `main.py` for model training and testing.
If you want to select metapaths with incremental metapath optimization, run `incremental_metapath_optimization.py` several times and select metapaths with the highest performance.

## Model performance
|  | **KNN** | **LR** | **MLP** | **GB** | **GCN** | **GAT** | **HAN** | ***MCGAT*** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **AUROC** | 0.6208 | 0.6762 | 0.6500 | 0.6940 | 0.7361 | 0.7447 | 0.7536 | **0.7586** |
| **AUPRC** | 0.6305 | 0.6755 | 0.6223 | 0.6967 | 0.7144 | 0.7116 | 0.7132 | **0.7240** |

![image](https://github.com/user-attachments/assets/a5490264-1867-486f-98d8-f481821b5d98)


---

## **📑 Summary**

**"인공지능으로 신약 개발"**

지난40년간 FDA에 승인된 약물 중 약 3분의 1이 천연물을 기반으로 만들어졌을 정도로, 천연물은 약물 소재로서 높은 잠재력을 가집니다. 그래서 어떤 **천연물이 어떤 질병에 효과가 있을지를 예측**하는 것 또한 매우 중요하지만 어려운 문제입니다.

저는 **메타패스(Metapath) 기반의 그래프 신경망(GNN)을 활용하여 천연물과 질병(표현형) 간의 연관성을 예측하는 모델**을 개발했습니다. 특히, 기존 모델들이 다양한 생물학적 정보의 상호작용을 충분히 반영하지 못하는 문제를 해결하기 위해, **교차 유형 동기화(Cross-type Synchronization) 기법과 점진적 메타패스 최적화(Incremental Metapath Optimization) 기법을 개발**했습니다. 이를 통해 **AUROC 0.7586, AUPRC 0.7240**의 더 정교한 예측에 성공했고, 파킨슨병과 연관된 새로운 천연물을 발굴하는 성과도 거두었습니다.

---

## **⭐ Key Task**

### **1️⃣ 천연물-질병 연관성 예측 모델 개발**

✅ **문제 정의**

- 기존 연구들은 특정 노드 유형(예: 천연물, 화합물, 질병)만 독립적으로 업데이트하거나, 특정 노드 유형을 우선적으로 반영하는 한계를 가짐
- 다양한 생물학적 관계를 활용하여 **천연물과 질병 간의 연관성을 더 정확하게 예측하는 모델이 필요**

✅ **해결 방법**

- **메타패스(Metapath) 기반의 그래프 신경망(GNN) 도입 → 풍부한 생물학적 관계 학습**
- **Cross-type Synchronization 기법 개발 → 교차 유형 동기화로 더 정교한 표현 학습**
    
    모든 노드 유형의 업데이트를 동기화 → 더 풍부한 정보 학습
    
    1. 모든 노드 유형을 차례로 업데이트
    2. 먼저 업데이트된 노드 유형이 다음 노드 유형의 업데이트에 입력값으로 사용되도록 함
    3. 업데이트 순서대로 사전학습되도록 함 (화합물 → 천연물 → 질병)
- **Incremental Metapath Optimization 개발 → 성능 향상을 극대화하는 최적의 메타패스 자동 선택**

✅ **성과**

- **AUROC 0.7586, AUPRC 0.7240**으로 7개의 기존 모델 대비 우수한 예측 성능 기록
- 기존 연구에서 다루지 않았던 **4종의 신규 파킨슨병 관련 천연물**을 발굴

---

### **2️⃣ 점진적 메타패스 최적화(Incremental Metapath Optimization) 개발**

✅ **문제 정의**

- 기존 연구에서는 **메타패스의 선택이 임의적이며**, 최적의 메타패스를 보장하지 못함
- 메타패스의 조합이 많아질수록 계산 비용이 증가하는 문제 발생

✅ **해결 방법**

- 초기 기본 메타패스(Base Metapath) 설정 후 **새로운 메타패스를 하나씩 추가하며 성능 비교 → 의미 있는 성능 향상을 제공하는 메타패스만 유지**
- 대응 표본 T 검증(Paired t-test)을 통해 **불필요한 메타패스 제외**

✅ **성과**

- **기존 방식보다 연관성 예측 성능이 지속적으로 향상됨**
- **불필요한 계산 비용을 줄이면서도 모델의 성능을 최적화**

---

## **👩‍🔧 Team**

- 단독 제 1저자 연구

---

## **💪 Contribution**

✅ **MSGAT (Metapath-based Synchronized Graph Attention Network) 새로운 모델 개발 및 최적화**

✅ **Cross-type Synchronization 기법 개발 → 노드 유형 간 정보 공유를 통한 예측 성능 개선**

✅ **Incremental Metapath Optimization 기법 개발 → 성능 향상을 극대화하는 최적의 메타패스 선택**

✅ **파킨슨병과 관련된 신규 천연물 4종 발굴 → 실제 생물학적 연구에 기여 가능성 제시**

✅ **MLP, GCN, GAT, HAN 등 기존 모델과의 비교 실험 수행 및 성능 평가**

---

![image](https://github.com/user-attachments/assets/fa67873d-60e2-42f0-a110-816fd90280ce)

