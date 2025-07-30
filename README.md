# DeepFaceDecoder
Implementation of the article: 'Deep Face Decoder: Towards understanding the embedding space of convolutional networks through visual reconstruction of deep face templates'.

## Estrutura

DeepFaceDecoder/
├── data/vggface2/          # Dataset
├── models/                 # Arquiteturas dos modelos
├── utils/                  # Utilitários
├── checkpoints/           # Modelos salvos
├── logs/                  # Logs do TensorBoard
├── config.py              # Configurações
├── train_dfd.py          # Script de treinamento
├── test_dfd.py           # Script de teste
└── inference.py          # Script de inferência

## Instalação

### 1. Clonar repositório
```
git clone https://github.com/seu-usuario/DeepFaceDecoder.git
cd DeepFaceDecoder
```

### 2. Ambiente virtual

```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Dependências

```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

### 4. Dataset

#### Download VGGFace2

1. Acesse: https://www.kaggle.com/datasets/hearfool/vggface2
2. Faça o download do dataset
3. Extraia o arquivo baixado

#### Organização dos Dados

Crie a estrutura de pastas no diretório do projeto:

```
mkdir -p data/vggface2/train
mkdir -p data/vggface2/test
```
Organize da seguinte maneira

data/vggface2/
├── train/
│   ├── n000001/
│   │   ├── 0001_01.jpg
│   │   └── ...
│   ├── n000002/
│   └── ...
└── test/
    ├── n000001/
    ├── n000002/
    └── ...
    
### 5. Treinamento

```
python train_dfd.py
```

