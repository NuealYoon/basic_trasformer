transformer with nn. and TorchText

## 1. 개요
 pytorch의 nn.Transformer와 TorchText를 이용해서 transformer 모델을 만들고, 다음 단어가 무엇인지 예측하는 모델을 학습시키는 코드 입니다.

ex)

입력: A + B + C 

출력: B + C + D
 
 
## 2. 코드 구조
코드는 5개의 구조로 되어 있으며 아래와 같습니다.
1. Define the model
  - Transformer model에 관련된 코드 입니다. 크게 encoder와 decoder로 되어 있습니다. 특히 encoder의 경우는 nn.TransformerEncoder와 nn.TransformerEncoderLayer를 사용하여 만들어진 코드 입니다.
2. Load and batch data
  - dataset에 있는 데이터 중 WikiText2 데이터셋을 사용합니다. 데이터를 batch단위로 불러오며, vocab에 등록된 단어가 없으면 unk로 변경시키는 코드입니다.
3. Initiate an instance
  - 하이퍼파라미터에 관련된 코드 입니다. 관련 수치 값들이 저장되어 있습니다.
5. Run the model
  - 모델에 데이터를 입력하여 학습 시키는 코드와 학습된 결과를 평가하는 코드가 있습니다.
7. Evaluation the best model on the test dataset
  - test 데이터셋으로 모델이 어느정도 성능을 낼 수 있는지 평가하는 코드입니다.

## 2. 코드 실행
python train.py // 학습 

## 3. 라이브러리 설치 
conda install pytorch -c pytorch
