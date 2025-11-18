# GNN 학습 환경 설정 가이드

이 폴더는 GNN (Graph Neural Network) 학습을 위한 Python 환경을 설정하는 모든 파일을 포함하고 있습니다.

---

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [파일 설명](#파일-설명)
3. [설정 방법](#설정-방법)
4. [문제 해결](#문제-해결)
5. [다음 단계](#다음-단계)

---

## 🚀 빠른 시작

### Windows 사용자 (권장)

**자동 설정 스크립트 사용:**

1. PowerShell을 **관리자 권한**으로 실행
2. 이 폴더로 이동:
   ```powershell
   cd "c:\Users\hippo\relbench\chorok\0. Setup"
   ```
3. 스크립트 실행:
   ```powershell
   .\create_gnn_env.ps1
   ```

**수동 설정:**
- [SETUP_CONDA.md](./SETUP_CONDA.md) 문서를 참조하세요.

---

## 📁 파일 설명

### 1. `create_gnn_env.ps1`
**용도:** GNN 학습 환경을 자동으로 설정하는 PowerShell 스크립트

**기능:**
- Conda 환경 `gnn_env` 생성 (Python 3.10)
- PyTorch (CPU 버전) 설치
- PyTorch Geometric 설치
- 필수 패키지 설치 (scipy, networkx, matplotlib 등)
- 설치 검증

**실행 조건:**
- Anaconda 또는 Miniconda가 이미 설치되어 있어야 함
- PowerShell 실행 정책 설정 필요 (관리자 권한)

**실행 방법:**
```powershell
.\create_gnn_env.ps1
```

**예상 소요 시간:** 5-10분

---

### 2. `SETUP_CONDA.md`
**용도:** Conda 환경 설정 전체 과정을 단계별로 설명하는 문서

**포함 내용:**
1. Miniconda/Anaconda 설치 방법
2. Conda 환경 생성 및 관리
3. PyTorch, PyTorch Geometric 설치
4. VSCode 연동 방법
5. 문제 해결 팁

**언제 사용:**
- 자동 스크립트가 작동하지 않을 때
- 단계별로 직접 설정하고 싶을 때
- 설치 과정을 이해하고 싶을 때

---

### 3. `setup_environment.ps1`
**용도:** Miniconda 초기 설정 스크립트 (첫 설치용)

**기능:**
- Miniconda 설치 확인
- Conda 초기화 (PowerShell 연동)

**언제 사용:**
- Conda가 아직 설치되지 않았을 때
- `conda` 명령어가 인식되지 않을 때

**실행 방법:**
```powershell
.\setup_environment.ps1
```

**주의사항:**
- 이 스크립트 실행 후 PowerShell을 재시작해야 함
- 재시작 후 `create_gnn_env.ps1` 실행

---

## 🛠 설정 방법

### 방법 1: 자동 설정 (권장)

#### 전제 조건
- [ ] Windows 10 이상
- [ ] PowerShell 5.1 이상
- [ ] 인터넷 연결
- [ ] 약 3GB 이상의 디스크 공간

#### 단계별 진행

**Step 1: Conda가 설치되어 있는지 확인**
```powershell
conda --version
```

✅ 버전이 표시되면 → **Step 3**으로 이동
❌ 오류가 발생하면 → **Step 2** 진행

---

**Step 2: Conda 초기 설정 (필요시)**
```powershell
# PowerShell을 관리자 권한으로 실행한 후:
cd "c:\Users\hippo\relbench\chorok\0. Setup"
.\setup_environment.ps1

# 완료 후 PowerShell을 닫고 다시 열기
```

---

**Step 3: GNN 환경 생성**
```powershell
# PowerShell에서:
cd "c:\Users\hippo\relbench\chorok\0. Setup"
.\create_gnn_env.ps1
```

**예상 출력:**
```
========================================
GNN Environment Setup and Package Installation
========================================

Step 0: Accepting Anaconda Terms of Service...
✓ Terms of Service accepted

Step 1: Creating gnn_env environment...
✓ gnn_env environment created

Step 2: Installing PyTorch (CPU version)...
This may take a few minutes...
✓ PyTorch installed

Step 3: Installing PyTorch Geometric...
✓ PyTorch Geometric installed

Step 4: Installing additional packages...
✓ Additional packages installed

Step 5: Verifying installation...

✓ PyTorch: 2.x.x
✓ PyTorch Geometric: 2.x.x

========================================
🎉 Installation Complete!
========================================
```

---

**Step 4: 환경 활성화**
```powershell
conda activate gnn_env
```

프롬프트가 `(gnn_env) PS C:\...`로 바뀌면 성공!

---

**Step 5: 설치 확인 (선택사항)**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

---

### 방법 2: 수동 설정

자세한 단계별 가이드는 [SETUP_CONDA.md](./SETUP_CONDA.md)를 참조하세요.

---

## ❓ 문제 해결

### 문제 1: `conda` 명령어를 인식하지 못함
**증상:**
```
conda : 용어 'conda'이(가) cmdlet, 함수, 스크립트 파일 또는 실행할 수 있는 프로그램 이름으로 인식되지 않습니다.
```

**해결 방법:**
1. Anaconda/Miniconda가 설치되어 있는지 확인
   - 설치 경로: `C:\Users\hippo\miniconda3` 또는 `C:\Users\hippo\anaconda3`

2. Conda 초기화 스크립트 실행:
   ```powershell
   .\setup_environment.ps1
   ```

3. PowerShell 재시작

4. 여전히 안 되면, Anaconda Prompt를 사용:
   - 시작 메뉴 → Anaconda Prompt (또는 Anaconda PowerShell Prompt)

---

### 문제 2: 스크립트 실행 정책 오류
**증상:**
```
이 시스템에서 스크립트를 실행할 수 없으므로 ... 파일을 로드할 수 없습니다.
```

**해결 방법:**
PowerShell을 **관리자 권한**으로 실행한 후:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

또는:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

---

### 문제 3: PyTorch Geometric 설치 실패
**증상:**
```
ERROR: Could not find a version that satisfies the requirement torch-geometric
```

**해결 방법 1: Conda로 재시도**
```powershell
conda activate gnn_env
conda install pyg -c pyg -y
```

**해결 방법 2: 버전 지정 설치**
```powershell
conda activate gnn_env
pip install torch-geometric==2.3.0
```

---

### 문제 4: GPU 사용하고 싶음
**현재 스크립트는 CPU 버전을 설치합니다.**

GPU 버전을 원하면 [SETUP_CONDA.md](./SETUP_CONDA.md)의 "GPU가 있다면" 섹션을 참조하세요.

**요약:**
```bash
# CUDA 11.8 버전 (GPU가 CUDA를 지원하는 경우)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pyg -c pyg -c conda-forge -y
```

---

### 문제 5: VSCode에서 Python 환경이 보이지 않음
**증상:**
VSCode의 Python interpreter 목록에 `gnn_env`가 없음

**해결 방법:**
1. VSCode에서 `Ctrl + Shift + P`
2. "Python: Select Interpreter" 입력
3. "Enter interpreter path..." 선택
4. 다음 경로 입력:
   ```
   C:\Users\hippo\miniconda3\envs\gnn_env\python.exe
   ```
   (또는 Anaconda 설치 경로에 맞게 수정)

---

### 문제 6: 설치 중 네트워크 오류
**증상:**
```
CondaHTTPError: HTTP 000 CONNECTION FAILED
```

**해결 방법:**
1. 인터넷 연결 확인
2. 방화벽/VPN 설정 확인
3. 재시도:
   ```powershell
   conda clean --all
   .\create_gnn_env.ps1
   ```

---

## 🎯 다음 단계

환경 설정이 완료되었다면, 다음 단계로 진행하세요:

### 1. 환경 활성화 확인
```powershell
conda activate gnn_env
```

### 2. Week 1 학습 시작
```powershell
cd "c:\Users\hippo\relbench\chorok\1. GNN"
python week1_gnn_training.py
```

### 3. 학습 로드맵 확인
[chorok/README.md](../README.md)에서 전체 학습 계획을 확인하세요.

---

## 📚 추가 자료

### Conda 기본 명령어
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

### Python 패키지 관리
```bash
# 패키지 설치
conda install <package_name>
# 또는
pip install <package_name>

# 패키지 업데이트
conda update <package_name>

# 패키지 제거
conda remove <package_name>
```

---

## 📞 도움이 필요하신가요?

1. **문서 확인:** [SETUP_CONDA.md](./SETUP_CONDA.md)의 "문제 해결" 섹션
2. **PyTorch 공식 문서:** https://pytorch.org/get-started/locally/
3. **PyG 공식 문서:** https://pytorch-geometric.readthedocs.io/
4. **Conda 공식 문서:** https://docs.conda.io/

---

## ✅ 설정 완료 체크리스트

설정이 완료되었다면 아래 항목을 확인하세요:

- [ ] Conda가 정상 작동함 (`conda --version`)
- [ ] `gnn_env` 환경이 생성됨 (`conda env list`)
- [ ] 환경 활성화 가능 (`conda activate gnn_env`)
- [ ] PyTorch 설치 확인 (`python -c "import torch; print(torch.__version__)"`)
- [ ] PyTorch Geometric 설치 확인 (`python -c "import torch_geometric; print(torch_geometric.__version__)"`)
- [ ] VSCode에서 Python interpreter로 `gnn_env` 선택 가능

모든 항목이 ✅ 체크되었다면, **Week 1 학습을 시작**하세요!

---

**생성일:** 2025-10-15
**최종 수정일:** 2025-10-15
