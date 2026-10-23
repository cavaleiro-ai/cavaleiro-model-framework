[English](README.md) · **Português (Brasil)**

<a id="top"></a>

<p align="center">
  <img src="docs/assets/cavaleiro-banner.svg" alt="Cavaleiro Model Framework" width="100%" />
</p>

# Cavaleiro Model Framework

<p align="center">
  <a href="https://github.com/cavaleiro-ai/cavaleiro-model-framework/actions/workflows/cavaleiro-ci.yml"><img alt="Cavaleiro CI" src="https://github.com/cavaleiro-ai/cavaleiro-model-framework/actions/workflows/cavaleiro-ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Licença: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="CHANGELOG.md"><img alt="Versão: Cavaleiro Model Framework 1.0.0" src="https://img.shields.io/badge/release-1.0.0-informational.svg"></a>
</p>

**O framework de definição de modelos para texto, visão, áudio, vídeo e multimodal — com correções verificadas de robustez e de Windows.**

O Cavaleiro Model Framework traz a biblioteca completa — centenas de arquiteturas de modelos, a API `pipeline`, o `generate`, tokenizadores e processadores, o `Trainer`, a CLI `transformers` — com correções de bugs no código central, que não depende de modelo específico: geração, pipelines, envio ao Hub, leitura de argumentos e a CLI de desenvolvimento. Cada correção tem um teste de regressão que falha no commit de base e passa aqui.

É um substituto direto: você continua usando `import transformers`, os modelos continuam vindo do [Hugging Face Hub](https://huggingface.co/models), e seu código e seus checkpoints continuam funcionando.

## O que o Cavaleiro Model Framework acrescenta

Tudo o que existe no projeto original no commit `d4804f9`, mais ([changelog completo](CHANGELOG.md)):

- **Geração que não quebra mais com `encoder_outputs`:** `encoder_repetition_penalty` e `encoder_no_repeat_ngram_size` são ignorados com um aviso, em vez de quebrar ou usar ids de preenchimento em silêncio, quando você passa saídas do encoder já calculadas.
- **Argumentos de `pipeline()` mais seguros:** `code_revision` funciona dentro de `model_kwargs` ou como argumento nomeado e chega ao carregamento do modelo, e `pipeline()` não escreve mais no dicionário `model_kwargs` que você passa, que pode ser reutilizado.
- **Nomes e caminhos corretos no `push_to_hub`:** o nome padrão do repositório funciona com `pathlib.Path`, com separador no fim do caminho e com barras normais no Windows, nas dez implementações de `save_pretrained`; arquivos de subpastas são enviados com caminhos `/` no Windows.
- **Correções para Windows:** o `HfArgumentParser` lê arquivos `.args` como UTF-8; o `transformers add-new-model-like` executa seus scripts auxiliares com o interpretador atual e funciona a partir de um ambiente virtual não ativado.
- **Compatibilidade restaurada:** `AutoVideoProcessor.register(config, processor, True)` volta a aceitar o `exist_ok` posicional antigo.
- **CI e documentação seguras para o fork:** um workflow [Cavaleiro CI](.github/workflows/cavaleiro-ci.yml) que só precisa de um token de leitura, os workflows de publicação e de GPU do projeto original [desativados com instruções para reativar](.github/workflows-disabled/README.md), [notas de engenharia](docs/cavaleiro/ENGINEERING.md) com o mapa da arquitetura e as evidências dos testes, e um [guia completo em português](docs/cavaleiro/GUIA-PT-BR.md).

## Instalação

O Cavaleiro Model Framework funciona com Python 3.10 a 3.14 e [PyTorch](https://pytorch.org/get-started/locally/) 2.6 ou mais recente. Ele é distribuído como código-fonte: não está publicado no PyPI, e `pip install transformers` instala o projeto original a partir do PyPI.

Clone o repositório e instale em um ambiente virtual. Com o [uv](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/cavaleiro-ai/cavaleiro-model-framework.git
cd cavaleiro-model-framework
uv venv --python 3.12 .venv
# Linux e macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

uv pip install -e ".[torch]"
```

Ou com o `venv` da biblioteca padrão e o `pip`:

```shell
python -m venv .venv
# ative como acima e depois:
pip install -e ".[torch]"
```

Para um PyTorch só de CPU no Linux, instale o PyTorch antes, pelo índice de CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. No Windows e no macOS, as wheels do PyTorch no PyPI já são de CPU (ou Apple Silicon). Para conferir a instalação, rode `transformers env`.

Os outros extras estão definidos no [`setup.py`](setup.py), por exemplo `".[torch,vision,sentencepiece]"`, ou `".[testing]"` para desenvolvimento.

## Primeiros passos

A API [`pipeline`](docs/source/en/pipeline_tutorial.md) é o jeito mais rápido de rodar um modelo. Ela baixa e guarda o modelo em cache, pré-processa a entrada e devolve uma saída legível:

```py
from transformers import pipeline

pipe = pipeline(task="text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
pipe("O segredo para fazer um bolo muito bom é ", max_new_tokens=40)
```

Converse com um modelo passando uma lista de mensagens:

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "Você é um assistente objetivo."},
    {"role": "user", "content": "Me dê três ideias para um fim de semana em São Paulo."},
]
pipe = pipeline(task="text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16, device_map="auto")
resposta = pipe(chat, max_new_tokens=256)
print(resposta[0]["generated_text"][-1]["content"])
```

Também dá para conversar pelo terminal enquanto o [`transformers serve`](docs/source/en/serve-cli/serving.md) estiver rodando: `transformers chat Qwen/Qwen2.5-0.5B-Instruct`.

<details>
<summary>Reconhecimento automático de fala</summary>

```py
from transformers import pipeline

pipe = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>Classificação de imagens</summary>

```py
from transformers import pipeline

pipe = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
# [{'label': 'macaw', 'score': 0.997...}, ...]
```

</details>

<details>
<summary>Perguntas sobre imagens</summary>

```py
from transformers import pipeline

pipe = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipe(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
# [{'answer': 'statue of liberty'}]
```

</details>

O uso de nível mais baixo não muda: `AutoTokenizer`, `AutoModelFor...`, `model.generate(...)`, `Trainer`. A [documentação da biblioteca](docs/source/en/index.md) vale para o Cavaleiro Model Framework como está; o [guia em português](docs/cavaleiro/GUIA-PT-BR.md) traz mais exemplos.

## Compatibilidade

O Cavaleiro Model Framework muda a identidade do projeto, não os identificadores de que o seu código depende:

- O pacote Python continua sendo **`transformers`**: `import transformers`, `from transformers import AutoModel` e todos os caminhos de import continuam iguais. A string de versão é a versão de desenvolvimento do projeto original em que esta versão se baseia (`5.21.0.dev0`).
- A ferramenta de linha de comando continua sendo **`transformers`** (`transformers env`, `transformers chat`, `transformers serve`, `transformers download`, `transformers add-new-model-like`).
- Variáveis de ambiente (`HF_HOME`, `HF_HUB_OFFLINE`, `HF_TOKEN`, `TRANSFORMERS_VERBOSITY`, ...), locais de cache, nomes de arquivos de configuração (`config.json`, `generation_config.json`, `tokenizer_config.json`, ...) e formatos de checkpoint não mudaram. Modelos salvos com o Cavaleiro Model Framework carregam no projeto original e vice-versa.
- Modelos, tokenizadores e datasets continuam vindo do Hugging Face Hub (`huggingface.co`), pelo `huggingface_hub`, como no projeto original.
- Como as duas distribuições se chamam `transformers`, instale só uma delas em cada ambiente.
- **Telemetria, igual à do projeto original:** as requisições ao Hub enviam um user agent com as versões da biblioteca, do Python e do PyTorch, um id de sessão aleatório e o contexto de uso (por exemplo, se foi usado um pipeline ou uma classe auto). Defina `HF_HUB_DISABLE_TELEMETRY=1` para enviar só as versões e `telemetry/off`, ou `HF_HUB_OFFLINE=1` para não fazer nenhuma requisição ao Hub.

### Observações para Windows

- O cache do Hugging Face Hub usa links simbólicos. Sem o Modo de Desenvolvedor (ou permissão de administrador), ele copia os arquivos e mostra um aviso; ative o Modo de Desenvolvedor para economizar espaço em disco.
- O `torch.compile` precisa do compilador C++ da Microsoft (`cl.exe`, das Build Tools do Visual Studio) no `PATH`. Sem ele, os trechos que compilam falham com `InvalidCxxCompiler`; todo o resto funciona.
- Defina `PYTHONUTF8=1` se os seus próprios scripts leem arquivos de texto sem informar a codificação.

## Como contribuir

Relatos de bugs, correções e melhorias na documentação são bem-vindos. Leia o [guia de contribuição](CONTRIBUTING.md) para preparar o ambiente e rodar os testes, e reporte vulnerabilidades em particular, como explica a [política de segurança](SECURITY.md). Novas arquiteturas de modelos e traduções da documentação estão fora do escopo deste repositório.

[Guia de contribuição](CONTRIBUTING.md) · [Política de segurança](SECURITY.md) · [Código de Conduta](CODE_OF_CONDUCT.md) · [Changelog](CHANGELOG.md) · [Notas de engenharia](docs/cavaleiro/ENGINEERING.md) · [Guia em português](docs/cavaleiro/GUIA-PT-BR.md) · [READMEs traduzidos](docs/original-i18n/README.md)

## Licença

Baseado no Transformers · licença Apache 2.0 · veja [LICENSE](LICENSE) e [NOTICE](NOTICE). Sem afiliação com o projeto original.

<p align="center"><a href="#top">Voltar ao topo ↑</a></p>
