# Conda 환경 설정 가이드

## 1단계: Anaconda 또는 Miniconda 설치

### 옵션 A: Miniconda (추천 - 가볍고 빠름)
1. [Miniconda 다운로드 페이지](https://docs.conda.io/en/latest/miniconda.html) 방문
2. Windows 64-bit installer 다운로드
3. 다운로드한 `.exe` 파일 실행
4. 설치 중 **"Add Anaconda to my PATH environment variable"** 옵션 체크 (권장)

### 옵션 B: Anaconda (전체 패키지 포함)
1. [Anaconda 다운로드 페이지](https://www.anaconda.com/download) 방문
2. Windows installer 다운로드
3. 설치 진행

---

## 2단계: Conda 환경 생성

설치가 완료되면 **Anaconda Prompt** 또는 **Anaconda PowerShell Prompt**를 열고 아래 명령어를 실행하세요:

```bash
# GNN 학습용 conda 환경 생성 (Python 3.10)
conda create -n gnn_env python=3.10 -y

# 환경 활성화
conda activate gnn_env
```

---

## 3단계: 필요한 패키지 설치

환경을 활성화한 상태에서 아래 명령어를 실행하세요:

```bash
# PyTorch 설치 (CPU 버전)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# PyTorch Geometric 설치
conda install pyg -c pyg -y

# 또는 pip로 설치
pip install torch-geometric

# 추가 유용한 패키지
conda install matplotlib numpy pandas jupyter -y
```

### GPU가 있다면 (CUDA 지원):
```bash
# PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# PyTorch Geometric with CUDA
conda install pyg -c pyg -c conda-forge -y
```

---

## 4단계: VSCode에서 Conda 환경 선택

### 방법 1: Command Palette 사용
1. VSCode를 열고 `Ctrl + Shift + P` 키를 누름
2. "Python: Select Interpreter" 입력 후 선택
3. 목록에서 `gnn_env` 환경을 찾아 선택
   - 예: `Python 3.10.x ('gnn_env')`

### 방법 2: 하단 상태바 클릭
1. VSCode 하단 우측의 Python 버전 클릭 (예: `Python 3.x.x`)
2. `gnn_env` 환경 선택

### 방법 3: settings.json에 직접 설정
1. `Ctrl + Shift + P` → "Preferences: Open Settings (JSON)"
2. 아래 내용 추가:
```json
{
    "python.defaultInterpreterPath": "C:\\Users\\hippo\\anaconda3\\envs\\gnn_env\\python.exe"
}
```
(경로는 실제 conda 설치 위치에 맞게 조정)

---

## 5단계: 설치 확인

Anaconda Prompt에서:
```bash
conda activate gnn_env
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

또는 VSCode 터미널에서 (환경이 활성화된 상태):
```bash
python week1_gnn_training.py
```

---

## 6단계: 코드 실행

이제 `week1_gnn_training.py`를 실행할 수 있습니다!

### VSCode에서 실행:
1. `week1_gnn_training.py` 파일 열기
2. 우측 상단 ▶️ 버튼 클릭 또는 `Ctrl + F5`

### 터미널에서 실행:
```bash
conda activate gnn_env
cd c:\Users\hippo\relbench\chorok
python week1_gnn_training.py
```

---

## 문제 해결

### conda 명령어가 인식되지 않는 경우:
1. Anaconda Prompt를 관리자 권한으로 실행
2. 다음 명령어 실행:
```bash
conda init powershell
conda init cmd.exe
```
3. 터미널을 재시작

### VSCode에서 conda 환경이 보이지 않는 경우:
1. VSCode Python 확장 설치 확인
2. VSCode 재시작
3. `Ctrl + Shift + P` → "Python: Select Interpreter" → "Enter interpreter path..." → conda python.exe 경로 직접 입력

### PyTorch Geometric 설치 오류:
```bash
# 대안: pip로 설치
pip install torch torchvision
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

---

## 환경 관리 명령어

```bash
# 환경 목록 확인
conda env list

# 환경 활성화
conda activate gnn_env

# 환경 비활성화
conda deactivate

# 설치된 패키지 확인
conda list

# 환경 삭제 (필요시)
conda env remove -n gnn_env
```

---

## 다음 단계

환경 설정이 완료되면:
1. ✅ `week1_gnn_training.py` 실행
2. ✅ Cora 데이터셋으로 GNN 학습
3. ✅ 85% 이상 test accuracy 달성 확인!

🎉 Happy coding!
