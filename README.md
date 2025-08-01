# 🚀 Instruções para Configuração do Ambiente

Crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente virtual no PowerShell:

```powershell
.\venv\Scripts\activate.ps1
```

> 💡 Se estiver usando o Prompt de Comando (CMD), use:
>
> ```cmd
> .\venv\Scripts\activate.bat
> ```

Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

É necessário ter o [Ollama](https://ollama.com/) instalado na sua máquina.

Baixe o modelo LLaMA 3 com o comando:

```bash
ollama pull llama3
```

Inicie o servidor Ollama:

```bash
ollama serve
```

> 🔄 Deixe esse comando rodando em um terminal separado.

Com tudo pronto, execute o app:

```bash
streamlit run app.py
```

Se tudo estiver correto, a aplicação será aberta automaticamente no navegador. Caso contrário, revise os passos e verifique se o Ollama está rodando corretamente.
