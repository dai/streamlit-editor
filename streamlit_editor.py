from dotenv import load_dotenv
import streamlit as st
from io import BytesIO
import os
import pickle
from streamlit_quill import st_quill
from dataclasses import dataclass, field
from typing import Optional
import pyperclip
import dspy
import requests
from os import getenv
from dsp import LM
from ratelimit import limits
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit_analytics2 as streamlit_analytics


class OpenRouterClient(LM):
    RL_CALLS=40
    RL_PERIOD_SECONDS=60
    def __init__(self, api_key=None, base_url="https://openrouter.ai/api/v1", model="meta-llama/llama-3-8b-instruct:free", extra_headers=None, **kwargs):
        self.api_key = api_key or getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.model = model
        self.extra_headers = extra_headers or {}
        self.history = []
        self.provider = "openai"
        self.kwargs = {'temperature': 0.0,
            'max_tokens': 150,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'n': 1}
        self.kwargs.update(kwargs)

    def _get_choice_text(choice):
        return choice["message"]["content"]

    def _get_headers(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        headers.update(self.extra_headers)
        return headers

    @limits(calls=RL_CALLS, period=RL_PERIOD_SECONDS)
    def basic_request(self, prompt: str, **kwargs):
        headers = self._get_headers()
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **kwargs
        }

        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response_data = response.json()
        print(response_data)

        self.history.append({
            "prompt": prompt,
            "response": response_data,
            "kwargs": kwargs,
        })

        return response_data

    def __call__(self, prompt, **kwargs):
        req_kwargs = self.kwargs

        if not kwargs:
            req_kwargs.update(kwargs)

        response_data = self.basic_request(prompt, **req_kwargs)
        completions = [choice["message"]["content"] for choice in response_data.get("choices", [])]
        return completions

# Initialize dspy client
load_dotenv()

# Dictionary to store available LM configurations
lm_configs = {}

# Configure OpenRouter LM if API key available
if "OPENROUTER_API_KEY" in os.environ:
    lm_configs['openrouter'] = OpenRouterClient(
        model=os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        api_base="https://openrouter.ai/api/v1"
    )

# Configure OpenAI LM if API key available
if "OPENAI_API_KEY" in os.environ:
    lm_configs['openai'] = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"]
    )

# Configure Deepseek LM if API key available
if "DEEPSEEK_API_KEY" in os.environ:
    lm_configs['deepseek'] = dspy.LM(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"]
    )

# Configure Gemini LM if API key available
if "GEMINI_API_KEY" in os.environ:
    lm_configs['gemini'] = dspy.LM(
        model="gemini/gemini-2.0-flash-exp",
        api_key=os.environ["GEMINI_API_KEY"]
    )

# Configure GitHub LM if credentials available
if "GITHUB_TOKEN" in os.environ:
    lm_configs['github'] = dspy.LM(
        model="openai/gpt-4o-mini",
        api_base="https://models.inference.ai.azure.com", 
        api_key=os.environ["GITHUB_TOKEN"]
    )

# Configure Ollama LM if model is specified
if "OLLAMA_MODEL" in os.environ:
    lm_configs['ollama'] = dspy.LM(
        model=f"ollama_chat/{os.environ['OLLAMA_MODEL']}",
        api_base="http://localhost:11434",
        api_key=""
    )

# Select default LM based on availability
default_lm = None
for lm_name in ['openrouter', 'openai', 'deepseek', 'gemini', 'github', 'ollama']:  # Updated priority order
    if lm_name in lm_configs:
        default_lm = lm_configs[lm_name]
        break

# Ensure LM is loaded at application start
if not dspy.settings.lm:
    if default_lm:
        dspy.settings.configure(
            lm=default_lm,
            max_requests_per_minute=15,
            trace=[]
        )
    else:
        st.error("No LLM configuration available. Please check your environment variables.")

# --- Data Models ---
@dataclass
class FeedbackItem:
    content: str
    reference_text: Optional[str] = None

@dataclass
class SummaryItem:
    title: str
    summary: str
    original_text: str
    start_index: int
    end_index: int

@dataclass
class WorkspaceStats:
    """Statistics for workspace analytics"""
    word_count: int = 0
    section_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Workspace:
    doc_content: Optional[str] = None
    ai_modified_text: Optional[str] = None
    feedback_items: list[FeedbackItem] = field(default_factory=list)
    document_summaries: list[SummaryItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None
    description: Optional[str] = None
    stats: WorkspaceStats = field(default_factory=WorkspaceStats)

# --- Workspace Management Functions ---
def create_new_workspace(name: Optional[str] = None) -> str:
    """Create a new workspace with a unique ID"""
    workspace_id = str(len(st.session_state.workspaces) + 1)
    workspace_name = name or f"Workspace {workspace_id}"
    
    new_workspace = Workspace(
        name=workspace_name,
        doc_content="",
        ai_modified_text=None,
        feedback_items=[],
        document_summaries=[]
    )
    
    st.session_state.workspaces[workspace_id] = new_workspace
    st.session_state.current_workspace_id = workspace_id
    save_state_to_disk()
    return workspace_id

def delete_workspace(workspace_id: str):
    """Delete a workspace"""
    if workspace_id in st.session_state.workspaces:
        del st.session_state.workspaces[workspace_id]
        if st.session_state.current_workspace_id == workspace_id:
            st.session_state.current_workspace_id = None
        save_state_to_disk()

def get_current_workspace() -> Optional[Workspace]:
    """Get the current workspace"""
    if st.session_state.current_workspace_id:
        return st.session_state.workspaces.get(st.session_state.current_workspace_id)
    return None

def switch_workspace(workspace_id: str):
    """Switch to a different workspace"""
    if workspace_id in st.session_state.workspaces:
        st.session_state.current_workspace_id = workspace_id
        save_state_to_disk()

# --- LLM Signatures ---
class ContentReviser(dspy.Signature):
    """Signature for content revision task"""
    context = dspy.InputField(desc="Optional context or theme for revision")
    guidelines = dspy.InputField(desc="Optional guidelines for revision")
    text = dspy.InputField(desc="Text to be revised")
    revised_content = dspy.OutputField(desc="Revised version of the input text")

class FeedbackGenerator(dspy.Signature):
    """Signature for feedback generation task"""
    text = dspy.InputField(desc="Text to generate feedback for")
    reference_text = dspy.InputField(desc="Optional specific text to focus feedback on")
    feedback = dspy.OutputField(desc="Generated feedback")

class SummaryGenerator(dspy.Signature):
    """Signature for summary generation task"""
    text = dspy.InputField(desc="Text to summarize")
    title = dspy.OutputField(desc="Section title")
    summary = dspy.OutputField(desc="Generated summary")

# --- LLM Functions ---
def generate_content_revision(text: str, context: Optional[str] = None, guidelines: Optional[str] = None) -> str:
    """Generate revised content using LLM"""
    try:
        # Ensure LM is loaded
        if not dspy.settings.lm:
            dspy.settings.configure(lm=default_lm)  # Use the configured default LM
        
        reviser = dspy.Predict(ContentReviser)
        result = reviser(
            text=text,
            context=context or "",
            guidelines=guidelines or ""
        )
        return result.revised_content
    except Exception as e:
        print(f"Error in content revision: {str(e)}")
        return text  # Return original text on error

def generate_feedback_revision(text: str, feedback_list: list[str]) -> str:
    """Generate revised content based on feedback items"""
    try:
        # Ensure LM is loaded
        if not dspy.settings.lm:
            dspy.settings.configure(lm=default_lm)  # Use the configured default LM
        
        # Combine feedback into guidelines
        guidelines = "\n".join([f"- {item}" for item in feedback_list])
        reviser = dspy.Predict(ContentReviser)
        result = reviser(
            text=text,
            context="Revise based on feedback",
            guidelines=guidelines
        )
        return result.revised_content
    except Exception as e:
        print(f"Error in feedback revision: {str(e)}")
        return text

def get_feedback_item(reference_text: Optional[str] = None) -> FeedbackItem:
    """Generate a feedback item using LLM"""
    try:
        # Ensure LM is loaded
        if not dspy.settings.lm:
            dspy.settings.configure(lm=default_lm)  # Use the configured default LM
        
        # Get current workspace
        current_workspace = get_current_workspace()
        if not current_workspace or not current_workspace.doc_content:
            return FeedbackItem(
                content="Unable to generate feedback: No document content available.",
                reference_text=reference_text
            )
        
        # Create the generator with the current LM
        generator = dspy.Predict(FeedbackGenerator)
        
        # Generate appropriate prompt based on whether reference text is provided
        if reference_text:
            result = generator(
                text=f"Generate a modification suggestion specifically for the following text within the full document context:\n\nFull Document:\n{current_workspace.doc_content}\n\nSelected Text:\n{reference_text}\n\nMake sure to only provide one concise modification suggestion, no original text.",
                reference_text=reference_text
            )
        else:
            result = generator(
                text=f"Generate a general document modification suggestion for the following text:\n\n{current_workspace.doc_content}\n\nMake sure to only provide one concise modification suggestion, no original text.",
                reference_text=""
            )
        
        # Create and return the feedback item
        return FeedbackItem(
            content=result.feedback,
            reference_text=reference_text
        )
    except Exception as e:
        print(f"Error generating feedback: {str(e)}")
        return FeedbackItem(
            content="Unable to generate feedback at this time.",
            reference_text=reference_text
        )

def generate_document_summary(text: str) -> list[SummaryItem]:
    """Generate document summaries using LLM"""
    try:
        # Split text into sections (simplified version)
        sections = split_into_sections(text)
        summaries = []
        
        generator = dspy.Predict(SummaryGenerator)
        
        for i, section_text in enumerate(sections):
            result = generator(text=section_text)
            start_idx = len(''.join(sections[:i]))
            end_idx = start_idx + len(section_text)
            
            summaries.append(SummaryItem(
                title=result.title,
                summary=result.summary,
                original_text=section_text,
                start_index=start_idx,
                end_index=end_idx
            ))
        
        # Update section count in the current workspace
        current_workspace = get_current_workspace()
        if current_workspace:
            current_workspace.document_summaries = summaries
            update_workspace_stats(current_workspace)
        
        return summaries
    except Exception as e:
        print(f"Error generating summaries: {str(e)}")
        return []

def regenerate_summary(text: str) -> str:
    """Regenerate a single summary using LLM"""
    try:
        generator = dspy.Predict(SummaryGenerator)
        result = generator(text=text)
        return result.summary
    except Exception as e:
        print(f"Error regenerating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def split_into_sections(text: str, min_section_length: int = 500) -> list[str]:
    """Split text into logical sections using Langchain's RecursiveCharacterTextSplitter"""
    # Initialize the recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=min_section_length,
        chunk_overlap=0,
        separators=["\n## ", "\n# ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n\n", "\n", " ", ""]
    )
    
    # First try to split by markdown headers
    sections = splitter.split_text(text)
    
    # If no sections were created (or only one large section), fall back to character-based splitting
    if len(sections) <= 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_section_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        sections = splitter.split_text(text)
    
    # Ensure we always return at least one section
    return sections if sections else [text]

# --- App Configuration ---
if 'read_mode' not in st.session_state:
    st.session_state.read_mode = False

st.set_page_config(
    page_title="Document Management with Generative AI",
    layout="wide",
    initial_sidebar_state="collapsed" if st.session_state.get('read_mode', False) else "expanded",
    menu_items={
        'Get Help': 'https://x.com/StockchatEditor',
        'Report a bug': "https://x.com/StockchatEditor",
    }
)

streamlit_analytics.start_tracking(load_from_json=".streamlit/analytics.json")

# --- Helper Functions ---
def load_document(file):
    """Load markdown document with empty document handling"""
    try:
        content = file.read().decode('utf-8')
        return content
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return ""  # Return empty string as fallback

def save_document(content: str) -> BytesIO:
    """Save document content to bytes"""
    byte_io = BytesIO()
    byte_io.write(content.encode('utf-8'))
    byte_io.seek(0)
    return byte_io

def validate_text_selection(full_text: str, selected_text: str) -> bool:
    """
    Validates if the selected text is actually part of the full document
    """
    return selected_text.strip() in full_text

def update_workspace_stats(workspace: Workspace) -> None:
    """Update workspace statistics"""
    if not workspace.doc_content:
        workspace.stats = WorkspaceStats()
        return

    # Calculate word count
    words = workspace.doc_content.split()
    word_count = len(words)

    # Get section count
    section_count = len(workspace.document_summaries)

    # Update stats
    workspace.stats = WorkspaceStats(
        word_count=word_count,
        section_count=section_count,
        last_updated=datetime.now()
    )

def save_state_to_disk():
    """Save all workspaces to disk"""
    state_data = {
        'workspaces': {
            workspace_id: {
                'doc_content': workspace.doc_content,
                'ai_modified_text': workspace.ai_modified_text,
                'feedback_items': [
                    {'content': item.content, 'reference_text': item.reference_text}
                    for item in workspace.feedback_items
                ],
                'document_summaries': [
                    {
                        'title': item.title,
                        'summary': item.summary,
                        'original_text': item.original_text,
                        'start_index': item.start_index,
                        'end_index': item.end_index
                    }
                    for item in workspace.document_summaries
                ],
                'name': workspace.name,
                'description': workspace.description,
                'created_at': workspace.created_at.isoformat(),
                'last_modified': workspace.last_modified.isoformat(),
                'stats': {
                    'word_count': workspace.stats.word_count,
                    'section_count': workspace.stats.section_count,
                    'last_updated': workspace.stats.last_updated.isoformat()
                }
            }
            for workspace_id, workspace in st.session_state.workspaces.items()
        },
        'current_workspace_id': st.session_state.current_workspace_id
    }
    
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/doc_state.pkl', 'wb') as f:
        pickle.dump(state_data, f)

def load_state_from_disk():
    """Load workspaces from disk"""
    try:
        with open('.streamlit/doc_state.pkl', 'rb') as f:
            state_data = pickle.load(f)
            
            workspaces = {}
            for workspace_id, workspace_data in state_data.get('workspaces', {}).items():
                stats_data = workspace_data.get('stats', {})
                workspaces[workspace_id] = Workspace(
                    doc_content=workspace_data.get('doc_content'),
                    ai_modified_text=workspace_data.get('ai_modified_text'),
                    feedback_items=[
                        FeedbackItem(
                            content=item['content'],
                            reference_text=item.get('reference_text')
                        )
                        for item in workspace_data.get('feedback_items', [])
                    ],
                    document_summaries=[
                        SummaryItem(
                            title=item['title'],
                            summary=item['summary'],
                            original_text=item['original_text'],
                            start_index=item['start_index'],
                            end_index=item['end_index']
                        )
                        for item in workspace_data.get('document_summaries', [])
                    ],
                    name=workspace_data.get('name'),
                    description=workspace_data.get('description'),
                    created_at=datetime.fromisoformat(workspace_data.get('created_at')),
                    last_modified=datetime.fromisoformat(workspace_data.get('last_modified')),
                    stats=WorkspaceStats(
                        word_count=stats_data.get('word_count', 0),
                        section_count=stats_data.get('section_count', 0),
                        last_updated=datetime.fromisoformat(stats_data.get('last_updated', workspace_data.get('created_at')))
                    )
                )
            
            st.session_state.workspaces = workspaces
            st.session_state.current_workspace_id = state_data.get('current_workspace_id')
    except FileNotFoundError:
        st.session_state.workspaces = {}
        st.session_state.current_workspace_id = None
    except Exception as e:
        st.error(f"Error loading state: {str(e)}")
        st.session_state.workspaces = {}
        st.session_state.current_workspace_id = None

def regenerate_document_from_summaries(summaries: list[SummaryItem]) -> str:
    """Reconstruct document from summaries"""
    return '\n\n'.join(item.original_text for item in summaries)

@st.fragment
def ai_assistant_column():
    current_workspace = get_current_workspace()
    if not current_workspace:
        return
    
    st.title("AI Assistant")
    tab1, tab2 = st.tabs(["All Feedback", "Custom Feedback"])
    
    with tab1:
        # Display existing feedback items
        if current_workspace.feedback_items:
            for idx, item in enumerate(current_workspace.feedback_items):
                cols = st.columns([0.1, 0.75, 0.15])
                
                # Checkbox column
                with cols[0]:
                    checkbox_key = f"feedback_checkbox_{idx}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = False
                    
                    # Update checkbox state
                    st.session_state[checkbox_key] = st.checkbox(
                        "Select feedback",
                        value=st.session_state[checkbox_key],
                        key=f"checkbox_display_{idx}",
                        label_visibility="collapsed"
                    )
                
                # Content column
                with cols[1]:
                    st.markdown(f"**{item.content}**")
                
                # Delete button column
                with cols[2]:
                    if st.button("🗑️", key=f"delete_feedback_{idx}"):
                        current_workspace.feedback_items.pop(idx)
                        # Clean up checkbox state
                        if f"feedback_checkbox_{idx}" in st.session_state:
                            del st.session_state[f"feedback_checkbox_{idx}"]
                        save_state_to_disk()
                        st.rerun()
                
                # Reference text expander
                if item.reference_text:
                    with st.expander("📌 View referenced text", expanded=False):
                        st.markdown(f"{item.reference_text}")
        else:
            st.info("No feedback items yet. Add feedback using the Custom Feedback tab.")
        
        # Get currently selected feedback items
        currently_selected_feedback = [
            item.content for idx, item in enumerate(current_workspace.feedback_items)
            if st.session_state.get(f"feedback_checkbox_{idx}", False)
        ]
        
        # Apply feedback button
        st.markdown("<hr class='custom-separator'>", unsafe_allow_html=True)
        if st.button(
            "✨ Apply Selected Feedback", 
            key="apply_feedback", 
            type="primary",
            disabled=not currently_selected_feedback,
            use_container_width=True
        ):
            try:
                if current_workspace.doc_content and currently_selected_feedback:
                    # Generate revised content
                    revised_text = generate_feedback_revision(
                        current_workspace.doc_content,
                        currently_selected_feedback
                    )
                    
                    # Update document content
                    current_workspace.doc_content = revised_text
                    
                    # Force Quill editor to update by changing its key
                    if 'quill_editor_key' not in st.session_state:
                        st.session_state.quill_editor_key = 0
                    st.session_state.quill_editor_key += 1
                    
                    # Get indices of selected feedback items to remove
                    indices_to_remove = [
                        idx for idx, item in enumerate(current_workspace.feedback_items)
                        if st.session_state.get(f"feedback_checkbox_{idx}", False)
                    ]
                    
                    # Remove the applied feedback items (in reverse order to maintain correct indices)
                    for idx in sorted(indices_to_remove, reverse=True):
                        current_workspace.feedback_items.pop(idx)
                        if f"feedback_checkbox_{idx}" in st.session_state:
                            del st.session_state[f"feedback_checkbox_{idx}"]
                    
                    # Save state and update UI
                    save_state_to_disk()
                    st.success("✅ Feedback applied successfully!")
                    st.rerun()
                else:
                    st.warning("Please select feedback to apply and ensure document has content.")
            except Exception as e:
                st.error(f"Error applying feedback: {str(e)}")
    
    with tab2:
        # Clipboard paste functionality
        if st.button(
            "📋 Paste Text from Clipboard",
            key="paste_clipboard",
            use_container_width=True
        ):
            selected_text = pyperclip.paste()
            if selected_text and current_workspace.doc_content and selected_text.strip() in current_workspace.doc_content:
                st.session_state.referenced_text = selected_text.strip()
            else:
                st.error("⚠️ Clipboard text was not found in the document.")
        
        # Show referenced text if it exists
        if hasattr(st.session_state, 'referenced_text') and st.session_state.referenced_text:
            st.info(f"📌 Selected text: '{st.session_state.referenced_text}'")
        
        # AI Feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🤖 Get General Feedback",
                key="get_general_feedback",
                type="secondary",
                use_container_width=True
            ):
                new_feedback = get_feedback_item()
                current_workspace.feedback_items.append(new_feedback)
                st.rerun(scope="fragment")
        
        with col2:
            if st.button(
                "🎯 Get Selected Text Feedback",
                key="get_selected_feedback",
                type="secondary",
                use_container_width=True,
                disabled=not hasattr(st.session_state, 'referenced_text') or not st.session_state.referenced_text
            ):
                new_feedback = get_feedback_item(reference_text=st.session_state.referenced_text)
                current_workspace.feedback_items.append(new_feedback)
                st.session_state.referenced_text = ""
                st.rerun(scope="fragment")
        
        st.markdown("<hr class='custom-separator'>", unsafe_allow_html=True)
        
        # Custom feedback input
        new_feedback = st.text_area(
            "Custom Feedback:", 
            value="",
            height=100,
            placeholder="Enter your feedback here...",
            key=f"feedback_input_{len(st.session_state.get('feedback_items', []))}"
        )
        
        # Add feedback button
        if st.button(
            "✍️ Add Custom Feedback",
            key="add_custom_feedback",
            type="primary",
            use_container_width=True
        ):
            if new_feedback.strip():
                new_item = FeedbackItem(
                    content=new_feedback.strip(),
                    reference_text=st.session_state.get('referenced_text', '')
                )
                current_workspace.feedback_items.append(new_item)
                st.session_state.referenced_text = ""
                st.rerun(scope="fragment")

# --- Session State ---
# Remove these legacy entries
if "doc_content" in st.session_state:
    del st.session_state.doc_content
if "ai_modified_text" in st.session_state:
    del st.session_state.ai_modified_text
if "feedback_items" in st.session_state:
    del st.session_state.feedback_items
if "document_summaries" in st.session_state:
    del st.session_state.document_summaries

# Initialize state
if "initialized" not in st.session_state:
    load_state_from_disk()
    st.session_state.initialized = True
    if 'quill_editor_key' not in st.session_state:
        st.session_state.quill_editor_key = 0

# Keep only workspace-related state
if "workspaces" not in st.session_state:
    st.session_state.workspaces = {}
if "current_workspace_id" not in st.session_state:
    st.session_state.current_workspace_id = None

# Add show_ai_assistant initialization
if "show_ai_assistant" not in st.session_state:
    st.session_state.show_ai_assistant = True
if "read_mode" not in st.session_state:
    st.session_state.read_mode = False
# Initialize current_summary_page
if "current_summary_page" not in st.session_state:
    st.session_state.current_summary_page = 0

# --- Sidebar ---
with st.sidebar:
    # AI Assistant toggle
    st.session_state.show_ai_assistant = st.toggle(
        "🤖 Show AI Assistant",
        value=st.session_state.show_ai_assistant
    )
    
    st.title("Workspace Manager")
    
    # Workspace Controls
    if st.button("➕ Create Workspace", 
                use_container_width=True,
                key="create_workspace_btn"):
        new_id = create_new_workspace()
        st.success(f"Created new workspace: {st.session_state.workspaces[new_id].name}")
        st.rerun()
    
    if st.session_state.workspaces:
        # Compact workspace selector with rename functionality
        workspace_names = {
            wid: ws.name for wid, ws in st.session_state.workspaces.items()
        }
        
        # Display current workspace name with edit button
        if st.session_state.current_workspace_id:
            current_ws = st.session_state.workspaces[st.session_state.current_workspace_id]
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                new_name = st.text_input(
                    "Workspace Name",
                    value=current_ws.name,
                    label_visibility="collapsed",
                    placeholder="Workspace name"
                )
                if new_name and new_name != current_ws.name:
                    current_ws.name = new_name
                    save_state_to_disk()
                    st.rerun()
            with col2:
                if st.button("✏️", help="Rename workspace"):
                    st.rerun()
        
        # Workspace switcher
        selected_workspace = st.selectbox(
            "Switch Workspace",
            options=list(workspace_names.keys()),
            format_func=lambda x: workspace_names[x],
            index=list(workspace_names.keys()).index(st.session_state.current_workspace_id)
            if st.session_state.current_workspace_id else 0,
            label_visibility="collapsed"
        )
        
        if selected_workspace != st.session_state.current_workspace_id:
            switch_workspace(selected_workspace)
            st.rerun()
    
    # Delete button
    if st.session_state.current_workspace_id:
        if st.button("🗑️ Delete Workspace", use_container_width=True):
            delete_workspace(st.session_state.current_workspace_id)
            st.rerun()
    
    st.markdown("<hr class='custom-separator'>", unsafe_allow_html=True)
    
    # Document Management Section (workspace-specific)
    current_workspace = get_current_workspace()
    if current_workspace:
        uploaded_file = st.file_uploader("Upload a Document", type=["md", "txt"])
        
        # Track file upload state
        if "last_uploaded_file" not in st.session_state:
            st.session_state.last_uploaded_file = None
        
        # Handle both upload and removal cases
        if uploaded_file is not None:
            # New file uploaded
            if st.session_state.last_uploaded_file != uploaded_file.name:
                try:
                    content = load_document(uploaded_file)
                    current_workspace.doc_content = content
                    current_workspace.ai_modified_text = None
                    current_workspace.document_summaries = []
                    update_workspace_stats(current_workspace)
                    
                    # Force Quill editor to update
                    st.session_state.quill_editor_key = st.session_state.get('quill_editor_key', 0) + 1
                    st.session_state.last_uploaded_file = uploaded_file.name
                    save_state_to_disk()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    current_workspace.doc_content = ""
        else:
            # File was removed
            if st.session_state.last_uploaded_file is not None:
                st.session_state.last_uploaded_file = None
                st.session_state.quill_editor_key = st.session_state.get('quill_editor_key', 0) + 1
                st.rerun()

        # Add separator
        st.markdown("<hr class='custom-separator'>", unsafe_allow_html=True)

        # Download Document button
        if current_workspace.doc_content:
            doc_name = current_workspace.name.replace(" ", "_").lower() if current_workspace.name else "document"
            st.download_button(
                "⬇️ Download Document",
                data=current_workspace.doc_content,
                file_name=f"{doc_name}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        # Add separator
        st.markdown("<hr class='custom-separator'>", unsafe_allow_html=True)
            
        if st.button("📊 Open Analytics Dashboard", 
            help="Access detailed usage analytics",
            use_container_width=True):
            st.query_params["analytics"] = "on"

# Update the styles
st.markdown("""
    <style>
    
    /* Reduce Sidebar width */
    [data-testid="stSidebar"] {
        width: 200px !important;
    }
    
    /* Expander styles */
    .streamlit-expanderHeader {
        width: 100%;
    }
    .streamlit-expanderContent {
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    
    /* Quill editor container */
    .element-container:has(> iframe) {
        overflow-y: scroll;
    }
    
    .stMainContainer {
        padding: 0rem !important;
        margin: 0rem !important;
    }

    .main .block-container {
        overflow-x: hidden !important;
    }

    /* Ensure Quill editor stays within bounds */
    iframe {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Read Mode Styles */
    .read-mode {
        max-width: 800px;
        margin: 1rem auto;
        font-family: 'Georgia', serif;
        line-height: 1.5;
        font-size: 18px;
        padding: 0rem 2rem;
    }
    
    /* Hide streamlit elements in read mode */
    .read-mode-active [data-testid="stToolbar"],
    .read-mode-active footer {
        display: none !important;
    }
    
    /* Custom separator with reduced spacing */
    .custom-separator {
        margin: 0.2rem 0 !important;
        border: none;
        border-top: 1px solid rgba(49, 51, 63, 0.2);
    }

    </style>
    """, unsafe_allow_html=True)

# --- Main App ---

current_workspace = get_current_workspace()

if current_workspace is None:
    st.write("<br>", unsafe_allow_html=True)
    st.info("Create a new workspace to get started.")
else:
    if st.session_state.read_mode:
        # Exit read mode button
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        if st.button("✕ Exit Read Mode", key="exit_read_mode", type="secondary", use_container_width=True):
            st.session_state.read_mode = False
            st.rerun()
        
        # Display content in read mode
        st.markdown('<div class="read-mode">', unsafe_allow_html=True)
        
        # Display the document content
        if current_workspace.doc_content:
            paragraphs = current_workspace.doc_content.split('\n')
            for paragraph in paragraphs:
                if paragraph.strip():  # Only display non-empty paragraphs
                    st.markdown(paragraph)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add class to body for read mode styles
        st.markdown("""
            <script>
                document.body.classList.add('read-mode-active');
            </script>
        """, unsafe_allow_html=True)
    else:
        # Create columns based on AI assistant visibility
        if st.session_state.show_ai_assistant:
            doc_col, ai_col = st.columns([0.7, 0.3])
        else:
            doc_col = st.container()
        
        # Document Editor Column
        with doc_col:
            st.title("Document Editor")
            
            # Create tabs for Document and Summary views
            doc_tab, summary_tab = st.tabs(["📄 Document", "📑 Summary"])
            
            with doc_tab:
                # Initialize content with empty string if None
                editor_content = current_workspace.doc_content if current_workspace.doc_content is not None else ""
                content = st_quill(
                    value=editor_content,
                    placeholder="Start writing...",
                    key=f"quill_editor_{st.session_state.get('quill_editor_key', 0)}"  # Dynamic key
                )
                
                if content != current_workspace.doc_content:
                    current_workspace.doc_content = content
                    update_workspace_stats(current_workspace)
                    save_state_to_disk()
            
            with summary_tab:
                if current_workspace.doc_content:
                    # Keep only the manual regenerate button
                    if st.button("🔄 Regenerate All Summaries", type="primary", use_container_width=True):
                        current_workspace.document_summaries = generate_document_summary(current_workspace.doc_content)
                        st.session_state.last_summary_content = current_workspace.doc_content
                        update_workspace_stats(current_workspace)
                        save_state_to_disk()
                        st.rerun()
                    
                    st.markdown("<hr class='custom-separator'>", unsafe_allow_html=True)
                    
                    # If no summaries exist yet, show a message
                    if not current_workspace.document_summaries:
                        st.info("Click 'Regenerate All Summaries' to generate section summaries.")
                    else:
                        # Only show pagination and tabs if we have summaries
                        # Pagination controls
                        items_per_page = 3
                        total_summaries = len(current_workspace.document_summaries)
                        total_pages = (total_summaries + items_per_page - 1) // items_per_page
                        current_page = st.session_state.get("current_summary_page", 0)

                        # Pagination UI
                        if total_pages > 1:
                            col_prev, col_page, col_next = st.columns([0.3, 0.4, 0.3])
                            with col_prev:
                                if st.button("⬅️ Previous", 
                                            disabled=(current_page <= 0), 
                                            key="prev_page",
                                            type="secondary",
                                            use_container_width=True):
                                    st.session_state.current_summary_page -= 1
                                    st.rerun()
                            with col_page:
                                st.markdown(
                                    f"<div style='text-align: center;'>Page {current_page + 1} of {total_pages}</div>", 
                                    unsafe_allow_html=True
                                )
                            with col_next:
                                if st.button("➡️ Next", 
                                            disabled=(current_page >= total_pages - 1), 
                                            key="next_page",
                                            type="secondary",
                                            use_container_width=True):
                                    st.session_state.current_summary_page += 1
                                    st.rerun()

                        # Get current page summaries
                        start_idx = current_page * items_per_page
                        end_idx = start_idx + items_per_page
                        current_summaries = current_workspace.document_summaries[start_idx:end_idx]

                        # Create tabs for current page summaries
                        section_tabs = st.tabs([item.title for item in current_summaries])

                        # Update all index references to use global_idx
                        for idx_in_page, (tab, summary_item) in enumerate(zip(section_tabs, current_summaries)):
                            global_idx = start_idx + idx_in_page
                            with tab:
                                # Section management buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("➕ Add Section Below", key=f"add_below_{global_idx}", use_container_width=True):
                                        new_section = SummaryItem(
                                            title=f"New Section {len(current_workspace.document_summaries) + 1}",
                                            summary="New section summary...",
                                            original_text="New section content...",
                                            start_index=summary_item.end_index,
                                            end_index=summary_item.end_index + 1
                                        )
                                        current_workspace.document_summaries.insert(global_idx + 1, new_section)
                                        update_workspace_stats(current_workspace)
                                        save_state_to_disk()
                                        st.rerun()
                                
                                with col2:
                                    if len(current_workspace.document_summaries) > 1:
                                        if st.button("🗑️ Delete Section", key=f"delete_{global_idx}", use_container_width=True):
                                            current_workspace.document_summaries.pop(global_idx)
                                            update_workspace_stats(current_workspace)
                                            save_state_to_disk()
                                            st.rerun()

                                # Summary section
                                summary_col, button_col = st.columns([0.85, 0.15])
                                with summary_col:
                                    edited_summary = st_quill(
                                        value=summary_item.summary,
                                        key=f"summary_quill_{global_idx}",
                                    )
                                    # Auto-save summary changes
                                    if edited_summary != summary_item.summary:
                                        summary_item.summary = edited_summary
                                        update_workspace_stats(current_workspace)
                                        save_state_to_disk()
                                
                                with button_col:
                                    if st.button(
                                        "🔄 Regenerate Summary",
                                        key=f"regenerate_summary_{global_idx}",
                                        help="Regenerate this summary from original section text",
                                        use_container_width=True
                                    ):
                                        new_summary = regenerate_summary(summary_item.original_text)
                                        current_workspace.document_summaries[global_idx].summary = new_summary
                                        update_workspace_stats(current_workspace)
                                        save_state_to_disk()
                                        st.rerun()
                                
                                # Update section button
                                if st.button("📝 Update Section Text from Summary", key=f"update_section_{global_idx}", use_container_width=True):
                                    try:
                                        new_section_text = generate_content_revision(
                                            summary_item.original_text,
                                            context=summary_item.summary,
                                            guidelines="Revise the text to better match the summary while maintaining the original content's essence."
                                        )
                                        current_workspace.document_summaries[global_idx].original_text = new_section_text
                                        new_doc_content = regenerate_document_from_summaries(current_workspace.document_summaries)
                                        current_workspace.doc_content = new_doc_content
                                        if 'quill_editor_key' not in st.session_state:
                                            st.session_state.quill_editor_key = 0
                                        st.session_state.quill_editor_key += 1
                                        update_workspace_stats(current_workspace)
                                        save_state_to_disk()
                                        st.success("✅ Section updated! Document content has been updated.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating section: {str(e)}")
                                
                                # Original text display
                                st.markdown("**Original Text:**")
                                paragraphs = summary_item.original_text.split('\n')
                                for paragraph in paragraphs:
                                    if paragraph.strip():
                                        st.markdown(paragraph)

        # AI Assistant Column (only shown if toggled on)
        if st.session_state.show_ai_assistant:
            with ai_col:
                ai_assistant_column()

# Display section count in the sidebar
def display_workspace_stats(workspace: Workspace):
    """Display workspace statistics in the sidebar"""
    st.sidebar.markdown("### 📊 Document Stats")
    
    # Display word and section count in columns
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Words", workspace.stats.word_count)
    with col2:
        st.metric("Sections", workspace.stats.section_count)
    
    # Display last edit time below
    last_edit = workspace.stats.last_updated
    time_str = last_edit.strftime("%I:%M %p")  # Format: HH:MM AM/PM
    date_str = last_edit.strftime("%b %d, %Y")  # Format: Month DD, YYYY
    
    st.sidebar.markdown(f"*Last edited: {time_str} on {date_str}*")

# Add the display call in the main app
if current_workspace:
    display_workspace_stats(current_workspace)

streamlit_analytics.stop_tracking(unsafe_password=os.environ.get("ANALYTICS_PASSWORD"), save_to_json=".streamlit/analytics.json")
