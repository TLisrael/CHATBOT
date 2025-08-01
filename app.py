import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from docx import Document
import re
from typing import Dict, Any, List
import requests
import io


class DocumentAnalyzer:
    """Classe para análisar  DOCX"""
    
    def __init__(self):
        pass
    
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Analisa um documento DOCX e extrai estilo de escrita"""
        try:
            doc = Document(file_path)
            
            full_text = []
            paragraphs_info = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
                    paragraphs_info.append({
                        'text': paragraph.text.strip(),
                        'word_count': len(paragraph.text.split())
                    })
            
            text_content = '\n'.join(full_text)
            
            sentences = re.split(r'[.!?]+', text_content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            words = text_content.split()
            unique_words = set(word.lower().strip('.,!?;:"()[]') for word in words)
            
            complex_sentences = [s for s in sentences if len(s.split()) > 20]
            simple_sentences = [s for s in sentences if len(s.split()) < 10]
            
            exclamations = text_content.count('!')
            questions = text_content.count('?')
            semicolons = text_content.count(';')
            colons = text_content.count(':')
            
            transition_words = ['além disso', 'portanto', 'entretanto', 'assim', 'consequentemente', 
                              'por outro lado', 'dessa forma', 'contudo', 'no entanto', 'ademais']
            transition_count = sum(text_content.lower().count(word) for word in transition_words)
            
            analysis = {
                'total_paragraphs': len(paragraphs_info),
                'total_words': len(words),
                'total_sentences': len(sentences),
                'unique_words': len(unique_words),
                'avg_sentence_length': len(words) / max(1, len(sentences)),
                'avg_paragraph_length': len(words) / max(1, len(paragraphs_info)),
                'vocabulary_richness': len(unique_words) / max(1, len(words)),
                'complex_sentences': len(complex_sentences),
                'simple_sentences': len(simple_sentences),
                'exclamations': exclamations,
                'questions': questions,
                'semicolons': semicolons,
                'colons': colons,
                'transition_words_count': transition_count,
                'sample_paragraphs': [p['text'] for p in paragraphs_info[:3]],
                'full_text': text_content,
                'sample_text': text_content[:2000] + "..." if len(text_content) > 2000 else text_content
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Erro ao analisar documento: {str(e)}")


class OllamaTextGenerator:
    """Gerador de texto  Ollama"""
    
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
    def check_ollama_connection(self):
        """Verifica se o Ollama está rodando"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self):
        """Obtém lista de modelos instalados no Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def generate_style_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
        """Cria prompt para análise de estilo"""
        return f"""
Analise o seguinte documento e descreva detalhadamente o estilo de escrita:

ESTATÍSTICAS DO DOCUMENTO:
- Total de parágrafos: {analysis['total_paragraphs']}
- Total de palavras: {analysis['total_words']}
- Total de sentenças: {analysis['total_sentences']}
- Tamanho médio das sentenças: {analysis['avg_sentence_length']:.1f} palavras
- Tamanho médio dos parágrafos: {analysis['avg_paragraph_length']:.1f} palavras
- Riqueza vocabular: {analysis['vocabulary_richness']:.2f}
- Sentenças complexas: {analysis['complex_sentences']}
- Sentenças simples: {analysis['simple_sentences']}
- Uso de pontuação: {analysis['exclamations']} exclamações, {analysis['questions']} perguntas
- Conectivos de transição: {analysis['transition_words_count']}

AMOSTRA DO TEXTO ORIGINAL:
{analysis['sample_text']}

PRIMEIROS PARÁGRAFOS:
{chr(10).join(analysis['sample_paragraphs'])}

Com base nessa análise, descreva em detalhes:
1. O tom e registro linguístico (formal/informal, técnico/coloquial, etc.)
2. A complexidade e estrutura das sentenças
3. O estilo do vocabulário utilizado
4. Como as ideias são organizadas e conectadas
5. Características distintivas do estilo de escrita
6. Padrões ret/óricos ou recursos estilísticos utilizados

Seja específico e detalhado na descrição do estilo.
"""
    
    def generate_content_prompt(self, style_description: str, topic: str) -> str:
        """Cria prompt para geração de conteúdo"""
        return f"""
Baseado na seguinte descrição de estilo de escrita, gere um texto completo sobre o tema "{topic}":

ESTILO A SER SEGUIDO:
{style_description}

INSTRUÇÕES ESPECÍFICAS:
1. Mantenha EXATAMENTE o mesmo tom e registro linguístico
2. Use estruturas de sentenças similares em complexidade 
3. Mantenha o mesmo padrão de vocabulário
4. Organize as ideias da mesma forma que o documento original
5. Use os mesmos tipos de conectivos e transições
6. Mantenha a mesma densidade de informação por parágrafo
7. O texto DEVE ter no minimo 800 e no maximo 1200 palavras
8. Inclua título e subtítulos se apropriado ao estilo original

TEMA: {topic}

Gere apenas o texto final, sem explicações adicionais ou comentários sobre o processo.
"""
    
    def call_ollama_api(self, prompt: str, max_tokens: int = 2000) -> str:
        """Chama a API do Ollama para gerar texto"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Erro na API Ollama: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Erro ao chamar Ollama: {str(e)}")
    
    def analyze_style(self, document_analysis: Dict[str, Any]) -> str:
        """Analisa o estilo do documento usando Ollama"""
        prompt = self.generate_style_analysis_prompt(document_analysis)
        return self.call_ollama_api(prompt, max_tokens=1000)
    
    def generate_content(self, style_description: str, topic: str) -> str:
        """Gera conteúdo baseado no estilo usando Ollama"""
        prompt = self.generate_content_prompt(style_description, topic)
        return self.call_ollama_api(prompt, max_tokens=2000)
    
    def chat_response(self, message: str, context: str = "") -> str:
        """Gera resposta para o chat"""
        system_context = """Você é um assistente especializado em análise de documentos e geração de conteúdo. 
        Responda de forma útil e informativa sobre análise de texto, estilos de escrita, e geração de conteúdo."""
        
        prompt = f"{system_context}\n\n{context}\n\nUsuário: {message}\n\nAssistente:"
        return self.call_ollama_api(prompt, max_tokens=500)


class DocumentGenerator:
    """Classe para gerar documentos DOCX"""
    
    def save_to_docx(self, content: str, output_path: str) -> bool:
        """Salva conteúdo em arquivo DOCX"""
        try:
            doc = Document()
            
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith('#'):
                        title_text = line.replace('#', '').strip()
                        level = 1 if line.startswith('# ') else 2
                        doc.add_heading(title_text, level=level)
                    elif line.isupper() and len(line.split()) <= 8:
                        doc.add_heading(line, level=1)
                    elif line.endswith(':') and len(line.split()) <= 6:
                        doc.add_heading(line.replace(':', ''), level=2)
                    else:
                        doc.add_paragraph(line)
            
            doc.save(output_path)
            return True
            
        except Exception as e:
            print(f"Erro ao salvar documento: {str(e)}")
            return False


def init_session_state():
    """Inicializa o estado da sessão do Streamlit"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_analyzed' not in st.session_state:
        st.session_state.documents_analyzed = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'style_description' not in st.session_state:
        st.session_state.style_description = None


def main():
    """Função principal da aplicação Streamlit"""
    
    st.set_page_config(
        page_title="IA Procedure Generator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header
    st.title("AI Procedure Generator")
    st.markdown("*Sistema inteligente*")
    
    with st.sidebar:
        st.header("Configurações")
        
        st.subheader(" Configuração Ollama")
        
        text_generator = OllamaTextGenerator()
        
        if text_generator.check_ollama_connection():
            st.success("Ollama conectado!")
            available_models = text_generator.get_available_models()
            
            if available_models:
                selected_model = st.selectbox(
                    "Modelo:",
                    available_models,
                    index=0 if "llama3.1" not in str(available_models) else 
                          [i for i, m in enumerate(available_models) if "llama3.1" in m][0] if any("llama3.1" in m for m in available_models) else 0
                )
                text_generator.model_name = selected_model
                st.info(f"Modelo selecionado: {selected_model}")
            else:
                st.warning("Nenhum modelo disponível")
                st.stop()
        else:
            st.error("Ollama não está rodando!")
            st.stop()
        
        st.divider()
        
        st.subheader("Configurações Avançadas")
        show_analysis = st.checkbox("Mostrar análise detalhada", value=True)
        show_style = st.checkbox("Mostrar descrição do estilo", value=True)
        
        st.divider()
        
        st.subheader("Histórico")
        if st.session_state.documents_analyzed:
            for i, doc in enumerate(st.session_state.documents_analyzed):
                with st.expander(f"📄 {doc['name']} - {doc['timestamp']}"):
                    st.write(f"**Tema gerado:** {doc['topic']}")
                    st.write(f"**Status:** {doc['status']}")
                    if 'word_count' in doc:
                        st.write(f"**Palavras geradas:** {doc['word_count']}")
        else:
            st.info("Nenhum documento analisado ainda")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Análise e Geração")
        
        uploaded_file = st.file_uploader(
            "Selecione um documento DOCX:",
            type=['docx'],
            help="Faça upload do documento que servirá como referência de estilo"
        )
        
        if uploaded_file:
            st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_input_path = tmp_file.name
            
            new_topic = st.text_input(
                "Tema para o novo texto:",
                placeholder="Ex: Escreva em português sobre Inteligência Artificial",
                help="Digite o tema sobre o qual você quer gerar um novo texto"
            )
            
            if st.button("Analisar Estilo e Gerar Texto", type="primary"):
                if new_topic:
                    try:
                        analyzer = DocumentAnalyzer()
                        doc_generator = DocumentGenerator()
                        
                        with st.spinner("📊 Analisando documento..."):
                            document_analysis = analyzer.analyze_document(temp_input_path)
                        
                        st.success("✅ Documento analisado!")
                        
                        if show_analysis:
                            with st.expander("📊 Análise Detalhada do Documento", expanded=False):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Total de Palavras", document_analysis['total_words'])
                                    st.metric("Total de Sentenças", document_analysis['total_sentences'])
                                    st.metric("Parágrafos", document_analysis['total_paragraphs'])
                                with col_b:
                                    st.metric("Tamanho Médio Sentença", f"{document_analysis['avg_sentence_length']:.1f}")
                                    st.metric("Riqueza Vocabular", f"{document_analysis['vocabulary_richness']:.2f}")
                                    st.metric("Sentenças Complexas", document_analysis['complex_sentences'])
                        
                        with st.spinner("Analisando estilo de escrita..."):
                            style_description = text_generator.analyze_style(document_analysis)
                            st.session_state.style_description = style_description
                        
                        st.success("stilo analisado!")
                        
                        if show_style:
                            with st.expander(" Descrição do Estilo Identificado", expanded=False):
                                st.write(style_description)
                        
                        with st.spinner("Gerando novo texto baseado no estilo de escrita..."):
                            generated_content = text_generator.generate_content(style_description, new_topic)
                        
                        st.success("Texto gerado!")
                        
                        with st.spinner("💾 Salvando documento..."):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"texto_gerado_{timestamp}.docx"
                            temp_output_path = os.path.join(tempfile.gettempdir(), output_filename)
                            
                            if doc_generator.save_to_docx(generated_content, temp_output_path):
                                st.success("Documento salvo!")
                            else:
                                st.error("Erro ao salvar documento")
                        
                        word_count = len(generated_content.split())
                        st.session_state.documents_analyzed.append({
                            'name': uploaded_file.name,
                            'topic': new_topic,
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'status': 'Concluído',
                            'output_path': temp_output_path,
                            'output_filename': output_filename,
                            'word_count': word_count
                        })
                        
                        st.session_state.current_analysis = {
                            'content': generated_content,
                            'output_path': temp_output_path,
                            'output_filename': output_filename,
                            'word_count': word_count
                        }
                        
                    except Exception as e:
                        st.error(f"Erro durante o processamento: {str(e)}")
                else:
                    st.warning("por favor, digite um tema para o novo texto")
        
        if st.session_state.current_analysis:
            st.subheader("Text//o Gerado")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Palavras Geradas", st.session_state.current_analysis['word_count'])
            with col_info2:
                if os.path.exists(st.session_state.current_analysis['output_path']):
                    with open(st.session_state.current_analysis['output_path'], 'rb') as file:
                        st.download_button(
                            label="Baixar Documento DOCX",
                            data=file.read(),
                            file_name=st.session_state.current_analysis['output_filename'],
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
            
            with st.expander(" Prévia do Texto Gerado", expanded=True):
                st.text_area(
                    "Conteúdo:",
                    value=st.session_state.current_analysis['content'],
                    height=400,
                    disabled=True
                )
    
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>IA Procedure Generator | Powered by WOOD</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
