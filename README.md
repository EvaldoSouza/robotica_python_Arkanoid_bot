# Agente de Aprendizado por Reforço para Arkanoid

Este projeto implementa um agente de Aprendizado por Reforço (RL - Reinforcement Learning) que aprende a jogar o clássico do NES, Arkanoid. Utilizando uma arquitetura baseada em Domain-Driven Design (DDD), o projeto conecta o emulador nes-py, um pipeline de visão computacional baseado em OpenCV e um algoritmo personalizado de Q-Learning para treinar uma Inteligência Artificial a dominar o jogo a partir dos pixels brutos da tela.

## Funcionalidades

*   **Pipeline de Visão Computacional Personalizado:** Extrai coordenadas sub-pixel, velocidades e trajetórias da bola diretamente do frame bruto do emulador, evitando hacks de injeção de RAM.
*   **Cérebro Q-Learning:** Implementa um simples algoritmo greedy de aprendizado por reforço.
*   **Dashboard de Telemetria ao Vivo:** Um painel em OpenCV rodando a 60 FPS mostrando o jogo, a representação de visão em cores falsas do agente e gráficos de valor-Q em tempo real.
*   **Métricas de Longo Prazo:** Integração com Matplotlib para rastrear recompensas episódicas, frames de sobrevivência e decaimento da exploração ao longo do tempo.

---

## Arquitetura e Estrutura Interna

O sistema foi redesenhado visando responsabilidade única e estabilidade no treinamento. Os componentes centrais englobam:

*   **Orquestração e Execução (`main.py`):** O `ArkanoidOrchestrator` executa o loop central interagindo com o emulador. Ele utiliza *Action Repeats* (atualizando as decisões do agente a cada 5 frames) e gerencia checkpoints de estado em memória para resetar a partida instantaneamente em caso de erro sem precisar recarregar todo o emulador.
*   **Cérebro RL (`rl/`):** 
    *   `ArkanoidBrain`: A fachada do agente, atuando com a política de escolha.
    *   `StateDiscretizer`: Reduz a complexidade da visão da tela transformando posições contínuas em *bins* de estado discretos baseados na distância em X entre a raquete e a trajetória da bola.
    *   `RewardShaper`: Fornece *sparse rewards*, pontuando rebotes na raquete (identificados pelas mudanças de vetores da bola) e penalizando severamente quedas (ausência de bola na tela).
*   **Renderização e Dashboard (`display/`):** Combinando visualização tática e estratégica. O `LiveRenderer` gera o display OpenCV em "tempo real", com gráficos de predição do valor-Q, enquanto o `MetricsRenderer` monitora e plota as curvas de sobrevivência e ganho de confiança do algoritmo usando Matplotlib.
*   **Persistência Seguro (`storage_gateway.py`):** Utiliza um padrão Gateway (`LocalDiskStorage`) para persistir automaticamente telemetria, configurações e o aprendizado contínuo (Matriz-Q e Epsilon) dentro de um diretório de sessão segura, garantindo que o treinamento possa ser pausado e retomado.

---

## Estrutura do Projeto

*   `main.py`: O orquestrador principal conectando todos os módulos do domínio.
*   `rl/`: Contém o Cérebro RL, a política TD(λ) implementada via Q-Learning, o discretizador de estado e o modelador de recompensa.
*   `vision/`: Ambiente de física baseado em OpenCV que rastreia a bola, a raquete e os blocos.
*   `emulator/`: Adaptador para a engine nes-py e tradutor de entrada via bitmask.
*   `display/`: Os componentes de interface (Dashboard ao vivo em OpenCV e métricas em Matplotlib).
*   `domain/`: Modelos de dados compartilhados (ex: FramePerception, Coordinate).

## Pré-requisitos

*   Python 3.12 (recomendado)
*   Um arquivo de ROM do Arkanoid para NES chamado `arkanoid.nes` colocado no diretório `roms/`.

## Instalação

Para evitar erros de conflito de versão (especialmente com a biblioteca gym), é altamente recomendado usar um ambiente virtual e instalar as dependências exatas fixadas no projeto.

Crie e ative um ambiente virtual:

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Instale as dependências exatas a partir do arquivo de requirements:

```bash
pip install -r requirements.txt
```

## Como Executar

O orquestrador é controlado pelo argumento `--mode`.

### 1. Modo de Demonstração (Showcase - Padrão)
Assista a um agente pré-treinado jogar de forma otimizada sem explorar ou atualizar sua tabela-Q.
```bash
# Executa usando um modelo no diretório raiz:
python main.py --mode showcase

# Executa usando o modelo campeão de uma sessão específica:
python main.py --mode showcase --session sessions/run_YYYYMMDD_HHMMSS
```

### 2. Modo de Treinamento com UI (Train UI)
Treine o agente do zero enquanto renderiza o jogo ao vivo, a visão do agente e as barras de decisão de valor-Q. Cria automaticamente uma pasta de sessão com data e hora.
```bash
python main.py --mode train_ui
```

### 3. Modo de Treinamento Sem Interface (Headless)
Treine o agente o mais rápido possível desativando a visualização do OpenCV. Exibirá telemetria no console e atualizará gráficos ao final dos episódios.
```bash
python main.py --mode train_headless
```

### 4. Retomando uma Sessão de Treinamento
Para continuar o treinamento de uma execução anterior, passe a flag `--session` apontando para o diretório específico gerado. Isso carrega de forma transparente a tabela-Q exata e a telemetria histórica para continuar de onde o agente parou.
```bash
python main.py --mode train_ui --session sessions/run_YYYYMMDD_HHMMSS
```

## Checkpoints e Persistência

O módulo RL salva automaticamente seu progresso na pasta da sessão ativa:
*   `arkanoid_brain.pkl`: O modelo sendo atualizado continuamente.
*   `arkanoid_best_brain.pkl`: O modelo "campeão", salvo quando o agente quebra seu próprio recorde de sobrevivência.
*   `telemetry_history.pkl` & `telemetry_raw.json`: Dados de métricas de treinamento.
*   `telemetry_chart.png`: Gráfico visual do progresso, atualizado automaticamente.

## Inspecionando o Cérebro do Agente

Se quiser ver os números brutos da matriz da tabela-Q e os hiperparâmetros (como exploração), use o script de inspeção autônomo.
```bash
python inspect_brain.py
```