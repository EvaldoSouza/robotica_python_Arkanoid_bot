Agente de Aprendizado por Reforço para Arkanoid

Este projeto implementa um agente de Aprendizado por Reforço (RL - Reinforcement Learning) que aprende a jogar o clássico do NES, Arkanoid. Utilizando uma arquitetura baseada em Domain-Driven Design (DDD), o projeto conecta o emulador nes-py, um pipeline de visão computacional baseado em OpenCV e um algoritmo personalizado de Q-Learning  para treinar uma Inteligência Artificial a dominar o jogo a partir dos pixels brutos da tela.

Funcionalidades

Pipeline de Visão Computacional Personalizado: Extrai coordenadas sub-pixel, velocidades e trajetórias da bola diretamente do frame bruto do emulador, evitando hacks de injeção de RAM.

Cérebro Q-Learning: Implementa um simple algoritmo greed de aprendizado por reforço

Dashboard de Telemetria ao Vivo: Um painel em OpenCV rodando a 60 FPS mostrando o jogo, a representação de visão em cores falsas do agente e gráficos de valor-Q em tempo real.

Métricas de Longo Prazo: Integração com Matplotlib para rastrear recompensas episódicas, frames de sobrevivência e decaimento da exploração ao longo do tempo.

Estrutura do Projeto

main.py: O orquestrador principal conectando todos os módulos do domínio.

rl/: Contém o Cérebro RL, a política TD(λ), o discretizador de estado e o modelador de recompensa.

vision/: Ambiente de física baseado em OpenCV que rastreia a bola, a raquete e os blocos.

emulator/: Adaptador para a engine nes-py e tradutor de entrada via bitmask.

display/: Os componentes de interface (Dashboard ao vivo em OpenCV e métricas em Matplotlib).

domain/: Modelos de dados compartilhados (ex: FramePerception, Coordinate).

Pré-requisitos

Python 3.12 (recomendado)

Um arquivo de ROM do Arkanoid para NES chamado arkanoid.nes colocado no diretório roms/.

Instalação

Para evitar erros de conflito de versão (especialmente com a biblioteca gym), é altamente recomendado usar um ambiente virtual e instalar as dependências exatas fixadas no projeto.

Crie e ative um ambiente virtual:

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate


Instale as dependências exatas a partir do arquivo de requirements:

pip install -r requirements.txt




Como Executar

O orquestrador é controlado pelo argumento --mode.

1. Modo de Demonstração (Showcase - Padrão)

Assista a um agente pré-treinado jogar de forma otimizada sem explorar ou atualizar sua tabela-Q.

# Executa usando um modelo no diretório raiz:
python main.py --mode showcase

# Executa usando o modelo campeão de uma sessão de treinamento específica:
python main.py --mode showcase --session sessions/run_YYYYMMDD_HHMMSS


2. Modo de Treinamento com UI (Train UI)

Treine o agente do zero enquanto renderiza o jogo ao vivo, a visão do agente e as barras de decisão de valor-Q. Cria automaticamente uma pasta de sessão com data e hora em sessions/.

python main.py --mode train_ui


3. Modo de Treinamento Sem Interface (Headless)

Treine o agente o mais rápido possível desativando a visualização do OpenCV. Ideal para deixar o agente aprendendo em segundo plano. Ele ainda exibirá a telemetria no console e atualizará os gráficos de métricas ao final dos episódios. Cria automaticamente uma pasta de sessão com data e hora em sessions/.

python main.py --mode train_headless


4. Retomando uma Sessão de Treinamento

Para continuar o treinamento de uma execução anterior (tanto no modo UI quanto no Headless), passe a flag --session apontando para o diretório específico gerado com a data e hora. Isso carrega de forma transparente a tabela-Q exata, a taxa de exploração (epsilon) e a telemetria histórica para continuar exatamente de onde o agente parou.

# Retomar com a interface gráfica (UI):
python main.py --mode train_ui --session sessions/run_YYYYMMDD_HHMMSS

# Retomar em modo headless:
python main.py --mode train_headless --session sessions/run_YYYYMMDD_HHMMSS


Checkpoints e Persistência

O módulo RL salva automaticamente seu progresso na pasta da sessão ativa:

arkanoid_brain.pkl: O modelo sendo atualizado continuamente.

arkanoid_best_brain.pkl: O modelo "campeão", salvo sempre que o agente quebra seu próprio recorde de sobrevivência.

telemetry_history.pkl & telemetry_raw.json: Os dados de métricas de treinamento a longo prazo.

telemetry_chart.png: Um gráfico visual do progresso do agente, atualizado automaticamente.

Inspecionando o Cérebro do Agente

Se você quiser ver os números brutos de toda a matriz da tabela-Q e os hiperparâmetros atuais (como a taxa de exploração), você pode usar o script de inspeção autônomo.

python inspect_brain.py
