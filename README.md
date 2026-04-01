# MEDIDOR-TEMPERATURA[README.md](https://github.com/user-attachments/files/26422095/README.md)
# 🌡️ Sistema de Monitoramento de Temperatura com IA

## 📌 Descrição do Projeto

O **Sistema de Monitoramento de Temperatura com IA** é uma solução que integra **hardware, software e inteligência artificial** para realizar a leitura, análise e visualização de temperatura em tempo real.

O sistema simula a coleta de dados de um sensor de temperatura, armazena essas informações em um banco de dados local e disponibiliza os dados por meio de um **dashboard interativo acessível via navegador**.

---

## ⚙️ Funcionalidades

- 📡 Monitoramento de temperatura em tempo real  
- 💾 Armazenamento de dados em banco SQLite  
- 📊 Visualização gráfica com atualização automática  
- 🤖 Previsão de temperatura utilizando Inteligência Artificial  
- 🌐 Interface web interativa  

---

## 🧠 Inteligência Artificial

O sistema utiliza um modelo simples de **regressão linear** para prever valores futuros de temperatura com base no histórico coletado, permitindo análise de tendências e comportamento térmico do ambiente.

---

## 🏗️ Arquitetura do Sistema

O fluxo do sistema é dividido em etapas:

1. Coleta de dados (sensor ou simulação)  
2. Armazenamento em banco de dados  
3. Processamento e análise dos dados  
4. Exibição em dashboard web  

---

## 🛠️ Tecnologias Utilizadas

- Python  
- Dash / Plotly (visualização)  
- SQLite (banco de dados)  
- Scikit-learn (Inteligência Artificial)  

---

## 🚀 Execução

Para executar o projeto:

```bash
pip install dash pandas plotly scikit-learn
python temp_monitor_ai_dashboard.py
```

Acesse no navegador:

```
http://127.0.0.1:8050
```

---

## 🎯 Objetivo

Demonstrar a aplicação de conceitos de:
- Sistemas digitais  
- Internet das Coisas (IoT)  
- Análise de dados  
- Inteligência Artificial  

em um sistema integrado e funcional.
