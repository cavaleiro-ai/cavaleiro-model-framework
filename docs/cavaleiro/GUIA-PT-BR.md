# Guia do Cavaleiro Model Framework (português)

Este guia explica o que é o Cavaleiro Model Framework, como ele é organizado, como instalar no Windows, no macOS e no Linux, como usar no dia a dia, como configurar, como resolver os problemas mais comuns e como contribuir. Para os detalhes técnicos das correções e dos testes, veja as [notas de engenharia](ENGINEERING.md) (em inglês).

## Sumário

1. [O que é e para quem](#1-o-que-é-e-para-quem)
2. [Arquitetura](#2-arquitetura)
3. [Instalação passo a passo](#3-instalação-passo-a-passo)
4. [Primeiros passos](#4-primeiros-passos)
5. [Exemplos práticos](#5-exemplos-práticos)
6. [Configuração](#6-configuração)
7. [Solução de problemas](#7-solução-de-problemas)
8. [Como contribuir](#8-como-contribuir)

## 1. O que é e para quem

O **Cavaleiro Model Framework** é a biblioteca que define e executa modelos de aprendizado de máquina de texto, visão, áudio, vídeo e multimodais, tanto para inferência quanto para treino. Ele traz a biblioteca completa do projeto original e acrescenta correções verificadas em partes centrais do código:

- geração de texto com saídas do encoder já calculadas (`encoder_outputs`);
- argumentos da função `pipeline()` (`model_kwargs` reutilizável e `code_revision`);
- nomes de repositório e caminhos de arquivo ao publicar no Hub com `push_to_hub`;
- leitura de arquivos `.args` em UTF-8 e a CLI `add-new-model-like` no Windows;
- compatibilidade de `AutoVideoProcessor.register`.

Cada correção tem um teste que falha no código original e passa aqui. A lista completa está no [CHANGELOG](../../CHANGELOG.md).

**Para quem é:**

- quem usa modelos do Hugging Face Hub em Python e quer as mesmas APIs com essas correções;
- quem desenvolve no **Windows**, onde várias das correções fazem diferença;
- quem quer estudar a arquitetura da biblioteca com um mapa em português.

**O que não muda:** o pacote continua se chamando `transformers` (você escreve `import transformers`), o comando continua sendo `transformers`, os modelos continuam vindo do Hugging Face Hub, e os arquivos salvos são compatíveis com o projeto original nos dois sentidos.

## 2. Arquitetura

### Visão geral

```text
                       seu código Python / comando `transformers`
                                        |
          +-----------------------------+------------------------------+
          |                                                            |
   pipeline("tarefa", model=...)                       AutoModel / AutoTokenizer / AutoProcessor
   (alto nível: tudo automático)                       (baixo nível: você controla cada etapa)
          |                                                            |
          +-----------------------------+------------------------------+
                                        v
                        Classes "Auto" (models/auto/)
                 model_type do config.json  ->  classe certa
                                        |
          +----------------+------------+--------------+------------------+
          v                v                           v                  v
   Configuração       Tokenizador /             Modelo (PyTorch)     GenerationConfig
   config.json        Processador               modeling_*.py        generation_config.json
   (hiperparâmetros)  (texto/imagem/áudio        + carregamento de     (como gerar texto)
                       -> tensores)              pesos safetensors
                                        |
                                        v
                    forward()  ou  generate()  (generation/)
                    processadores de logits, critérios de parada, cache KV
                                        |
                                        v
                    pós-processamento -> resultado legível

   Hub da Hugging Face  <---- download / cache (utils/hub.py) ---->  disco local
                        <---- save_pretrained(push_to_hub=True) --
```

### Componentes principais

| Componente | Onde fica | Para que serve |
| --- | --- | --- |
| Raiz do pacote | `src/transformers/__init__.py` | Expõe a API pública e importa os módulos sob demanda (import "preguiçoso"), para o `import transformers` ser rápido. |
| Configuração | `configuration_utils.py` | Guarda os hiperparâmetros do modelo e lê/grava o `config.json`. |
| Modelos | `modeling_utils.py` e `models/<nome>/` (mais de 500 arquiteturas) | Cada arquitetura num arquivo próprio; a classe base cuida de carregar, salvar, distribuir em dispositivos e publicar no Hub. |
| Classes Auto | `models/auto/` | Descobrem a classe certa a partir do `model_type`; aceitam classes personalizadas com `register()` e código remoto com `trust_remote_code=True`. |
| Tokenizadores | `tokenization_utils_base.py` e backends | Transformam texto em ids e de volta; padding, truncamento e templates de chat. |
| Processadores | `processing_utils.py`, `image_processing_*`, `video_processing_utils.py`, `feature_extraction_*` | Preparam imagens, vídeo e áudio, e combinam tudo com o tokenizador. |
| Geração | `generation/` | `generate()`: busca gulosa, amostragem, beam search, decodificação assistida, processadores de logits, streamers. |
| Pipelines | `pipelines/` | A função `pipeline()` e uma classe por tarefa (geração de texto, classificação, fala, imagem...). |
| Treino | `trainer.py`, `training_args.py`, `hf_argparser.py` | O `Trainer`, seus argumentos e a leitura de argumentos de linha de comando. |
| Integrações | `integrations/`, `quantizers/` | Kernels de atenção, DeepSpeed, FSDP, PEFT, paralelismo de tensores e dezenas de métodos de quantização. |
| Hub | `utils/hub.py` | Download, cache e publicação (`push_to_hub`). |
| CLI | `cli/` | `transformers env`, `chat`, `serve`, `download`, `add-new-model-like`. |

### Pontos de extensão

- **Modelos próprios:** registre com `AutoConfig.register(...)` e `AutoModel.register(...)`, ou publique no Hub com `auto_map` e carregue com `trust_remote_code=True`.
- **Pipelines próprios:** `PIPELINE_REGISTRY.register_pipeline(...)`.
- **Atenção:** `AttentionInterface.register(nome, função)`.
- **Geração:** passe `logits_processor=` e `stopping_criteria=` para `generate()`.
- **Treino:** callbacks com `TrainerCallback`.

### Testes e CI

Os testes ficam em `tests/` (um diretório por modelo em `tests/models/`, utilidades em `tests/utils/`, geração em `tests/generation/`, pipelines em `tests/pipelines/`). Testes marcados `@slow` usam modelos reais e só rodam com `RUN_SLOW=1`. No GitHub, o workflow [Cavaleiro CI](../../.github/workflows/cavaleiro-ci.yml) roda o ruff e os testes dos módulos alterados neste fork; os workflows do projeto original que publicam pacotes ou usam GPUs próprias foram [desativados](../../.github/workflows-disabled/README.md).

## 3. Instalação passo a passo

Requisitos: **Python 3.10 a 3.14** (recomendado: 3.12), **Git** e uns 3 GB livres para o PyTorch e as dependências. O Cavaleiro Model Framework é instalado a partir do código-fonte; ele não está no PyPI (lá, `pip install transformers` instala o projeto original).

### Windows

1. Instale o [Python 3.12](https://www.python.org/downloads/windows/) marcando **Add python.exe to PATH**, e o [Git for Windows](https://git-scm.com/download/win).
2. (Recomendado) Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/): no PowerShell, `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
3. (Recomendado) Ative o **Modo de Desenvolvedor** do Windows (Configurações → Sistema → Para desenvolvedores) para o cache do Hugging Face Hub poder usar links simbólicos.
4. Clone e instale, no PowerShell:

   ```powershell
   git clone https://github.com/cavaleiro-ai/cavaleiro-model-framework.git
   cd cavaleiro-model-framework
   git config core.autocrlf input
   uv venv --python 3.12 .venv
   .venv\Scripts\Activate.ps1
   uv pip install -e ".[torch]"
   ```

   Se o PowerShell bloquear o script de ativação, rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` uma vez, ou use diretamente `.venv\Scripts\python.exe`.
5. Confira: `transformers env`.

O PyTorch do PyPI para Windows é de CPU. Para usar uma GPU NVIDIA, instale antes o PyTorch com CUDA pelo seletor de [pytorch.org](https://pytorch.org/get-started/locally/) (por exemplo `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`).

### macOS

1. Instale o Python 3.12 (pelo [python.org](https://www.python.org/downloads/macos/) ou `brew install python@3.12`) e o Git (`xcode-select --install`).
2. Instale o uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
3. Clone e instale:

   ```bash
   git clone https://github.com/cavaleiro-ai/cavaleiro-model-framework.git
   cd cavaleiro-model-framework
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install -e ".[torch]"
   transformers env
   ```

Em Macs com Apple Silicon, o PyTorch usa a GPU pelo backend `mps` (`device="mps"` ou `device_map="auto"`).

### Linux

1. Instale Python 3.12, `git` e o venv pelo gerenciador da distribuição (por exemplo, `sudo apt install python3.12 python3.12-venv git`), ou use o uv para baixar o Python.
2. Instale o uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
3. Clone e instale:

   ```bash
   git clone https://github.com/cavaleiro-ai/cavaleiro-model-framework.git
   cd cavaleiro-model-framework
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   # Só CPU (download bem menor):
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install -e ".[torch]"
   transformers env
   ```

   Com GPU NVIDIA, pule a linha do índice de CPU: o PyTorch padrão do PyPI para Linux já vem com CUDA.

### Extras úteis

| Extra | Para quê |
| --- | --- |
| `torch` | PyTorch e Accelerate (necessário para rodar modelos) |
| `vision` | Imagens (`Pillow`, `torchvision`) |
| `audio` | Áudio (`librosa`, `torchaudio`, ...) |
| `sentencepiece` | Tokenizadores SentencePiece (T5, Llama antigos, ...) |
| `testing` | Ferramentas de teste e qualidade, para contribuir |

Combine assim: `uv pip install -e ".[torch,vision,sentencepiece]"`.

## 4. Primeiros passos

### Seu primeiro pipeline

```py
from transformers import pipeline

classificador = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(classificador("I love this library!"))
# [{'label': 'POSITIVE', 'score': 0.99...}]
```

Na primeira vez, o modelo é baixado e guardado no cache (`~/.cache/huggingface/hub`, ou `C:\Users\<você>\.cache\huggingface\hub` no Windows). Das próximas vezes, ele é lido do disco.

### Gerar texto

```py
from transformers import pipeline

gerador = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
mensagens = [{"role": "user", "content": "Explique em duas frases o que é um transformer."}]
saida = gerador(mensagens, max_new_tokens=80)
print(saida[0]["generated_text"][-1]["content"])
```

### Controlar cada etapa

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

modelo_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizador = AutoTokenizer.from_pretrained(modelo_id)
modelo = AutoModelForCausalLM.from_pretrained(modelo_id, dtype=torch.float32)

entradas = tokenizador.apply_chat_template(
    [{"role": "user", "content": "Qual é a capital do Brasil?"}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
ids = modelo.generate(**entradas, max_new_tokens=30, do_sample=False)
print(tokenizador.decode(ids[0][entradas["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Conversar pelo terminal

```bash
transformers serve          # em um terminal
transformers chat Qwen/Qwen2.5-0.5B-Instruct   # em outro
```

## 5. Exemplos práticos

### Reutilizar o mesmo `model_kwargs` em vários pipelines

Corrigido no Cavaleiro Model Framework: o dicionário que você passa não é mais alterado, então pode ser reutilizado.

```py
import torch
from transformers import pipeline

opcoes = {"low_cpu_mem_usage": True}
p1 = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", model_kwargs=opcoes, dtype=torch.float32)
p2 = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", model_kwargs=opcoes, dtype=torch.float32)
print(opcoes)  # {'low_cpu_mem_usage': True}
```

### Fixar a revisão do código remoto

Para modelos com código próprio no Hub (`trust_remote_code=True`), fixe a revisão do repositório e, se o código estiver em outro repositório, também a do código:

```py
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="organizacao/modelo-com-codigo",
    revision="<hash do commit do modelo>",
    code_revision="<hash do commit do código>",   # ou model_kwargs={"code_revision": ...}
    trust_remote_code=True,
)
```

Leia sempre o código antes de usar `trust_remote_code=True`.

### Usar saídas do encoder já calculadas

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
modelo = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
entrada = tok("translate English to German: The house is wonderful.", return_tensors="pt")

with torch.no_grad():
    saidas_encoder = modelo.get_encoder()(**entrada)

# Com só as saídas do encoder, a penalidade do encoder é ignorada com um aviso (antes, quebrava).
# Para aplicá-la, passe também os input_ids:
ids = modelo.generate(
    input_ids=entrada["input_ids"], encoder_outputs=saidas_encoder,
    encoder_repetition_penalty=1.2, max_new_tokens=20,
)
print(tok.decode(ids[0], skip_special_tokens=True))
```

### Salvar e publicar no Hub

```py
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

modelo = AutoModel.from_pretrained("google-bert/bert-base-uncased")
tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

destino = Path("saidas") / "meu-bert"        # Path, barras normais e barra final funcionam
modelo.save_pretrained(destino)
tok.save_pretrained(destino)
# Para publicar (precisa de login: `hf auth login`), o nome padrão do repositório é "meu-bert":
# modelo.save_pretrained(destino, push_to_hub=True)
```

### Argumentos de treino em arquivo `.args`

```py
# treino.py
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class Args:
    nome_execucao: str = "teste"
    epocas: int = 1

(args,) = HfArgumentParser(Args).parse_args_into_dataclasses(args_filename="treino.args")
print(args)
```

Com `treino.args` contendo `--nome_execucao Ação-de-teste --epocas 3`, o texto é lido em UTF-8 em qualquer sistema (antes, no Windows, virava `AÃ§Ã£o-de-teste`).

## 6. Configuração

### Variáveis de ambiente mais usadas

| Variável | Efeito |
| --- | --- |
| `HF_HOME` | Pasta base do Hugging Face Hub (cache, token). Padrão: `~/.cache/huggingface`. |
| `HF_HUB_CACHE` | Só a pasta do cache de modelos. |
| `HF_TOKEN` | Token de acesso ao Hub (para repositórios privados ou restritos). Prefira `hf auth login`. |
| `HF_HUB_OFFLINE=1` | Não acessa a internet; usa só o cache. |
| `HF_HUB_DISABLE_TELEMETRY=1` | Reduz o user agent enviado ao Hub (veja abaixo). |
| `TRANSFORMERS_VERBOSITY` | Nível de log: `debug`, `info`, `warning` (padrão), `error`. |
| `PYTHONUTF8=1` | (Windows) Faz o Python usar UTF-8 por padrão nos seus scripts. |

No PowerShell: `$env:HF_HUB_OFFLINE = "1"`. No bash: `export HF_HUB_OFFLINE=1`.

### Telemetria

Como no projeto original, as requisições ao Hub enviam um user agent com as versões da biblioteca, do Python e do PyTorch, um id de sessão aleatório e o contexto de uso (por exemplo, se foi usado um pipeline). Com `HF_HUB_DISABLE_TELEMETRY=1`, vão só as versões e `telemetry/off`; com `HF_HUB_OFFLINE=1`, nada é enviado.

### Arquivos de configuração de um modelo

| Arquivo | Conteúdo |
| --- | --- |
| `config.json` | Arquitetura e hiperparâmetros |
| `generation_config.json` | Padrões de geração (`max_new_tokens`, `temperature`, ...) |
| `tokenizer_config.json`, `tokenizer.json` | Tokenizador e template de chat |
| `preprocessor_config.json`, `processor_config.json` | Processadores de imagem, vídeo e áudio |
| `model.safetensors` (ou `model-0000X-of-0000Y.safetensors`) | Pesos |

## 7. Solução de problemas

| Sintoma | Causa provável | Solução |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'torch'` | PyTorch não instalado no ambiente ativo | Ative o `.venv` e rode `uv pip install -e ".[torch]"`. |
| `transformers` não é reconhecido como comando | Ambiente virtual não ativado | Ative (`.venv\Scripts\Activate.ps1` ou `source .venv/bin/activate`) ou use `python -m transformers.cli.transformers`. |
| Aviso sobre links simbólicos no cache (Windows) | Sem Modo de Desenvolvedor | Ative o Modo de Desenvolvedor; ou ignore (o cache copia os arquivos e ocupa mais espaço). |
| `InvalidCxxCompiler: Compiler: cl is not found` | `torch.compile` precisa do compilador C++ da Microsoft | Instale as Build Tools do Visual Studio (C++) ou não use `torch.compile`. |
| `PermissionError: [WinError 5]` ao apagar uma pasta com um modelo | O modelo carregado ainda mapeia os arquivos `.safetensors` na memória | Apague o objeto (`del modelo`) e rode `gc.collect()` antes de remover a pasta. |
| `OSError: You are trying to access a gated repo` | Modelo com acesso restrito | Aceite os termos na página do modelo e faça `hf auth login`. |
| `OSError: ... is not a valid model identifier` | Nome errado, sem internet, ou modelo privado | Confira o nome, a conexão e o token; ou use `HF_HUB_OFFLINE=1` com o modelo já em cache. |
| Falta de memória ao carregar | Modelo grande demais para a RAM/VRAM | Use um modelo menor, `dtype=torch.bfloat16`, `device_map="auto"`, ou quantização. |
| Texto com acentos estragado ao ler seus próprios arquivos (Windows) | Codificação padrão cp1252 | Use `encoding="utf-8"` ao abrir arquivos ou defina `PYTHONUTF8=1`. |
| `add-new-model-like` não gera o `modeling_*.py` | (corrigido aqui) scripts auxiliares rodavam com outro Python | Atualize para o Cavaleiro Model Framework 1.0.0; garanta `libcst` e `ruff` instalados (`".[quality]"`). |

Para relatar um problema, rode `transformers env` e use o [formulário de bug](https://github.com/cavaleiro-ai/cavaleiro-model-framework/issues/new/choose). Vulnerabilidades vão em particular, pela [política de segurança](../../SECURITY.md).

## 8. Como contribuir

1. Leia o [guia de contribuição](../../CONTRIBUTING.md) (em inglês) e o [Código de Conduta](../../CODE_OF_CONDUCT.md).
2. Faça um fork, crie um branch e instale com `uv pip install -e ".[testing,torch,vision,sentencepiece]"`.
3. Para um bug: escreva primeiro um teste que falha, depois a correção mínima que faz o teste passar.
4. Rode os testes do módulo que você alterou e `tests/utils` se mexeu em código compartilhado:

   ```bash
   python -m pytest -n 4 --dist=loadfile tests/utils
   ruff check <arquivos alterados>
   ruff format --check <arquivos alterados>
   ```

5. Na primeira vez que alterar um arquivo de código, acrescente abaixo do cabeçalho de licença uma linha `# Modified by Cavaleiro AI (<ano>): <o que mudou>` (exigência da Licença Apache 2.0).
6. Abra o pull request preenchendo o modelo e acrescente uma linha em **Unreleased** no [CHANGELOG](../../CHANGELOG.md).

Novas arquiteturas de modelos e traduções da documentação estão fora do escopo deste repositório.

---

Cavaleiro Model Framework é licenciado sob a [Apache 2.0](../../LICENSE). Veja o [NOTICE](../../NOTICE) e a seção [Licença](../../README.pt-BR.md#licença) do README.
