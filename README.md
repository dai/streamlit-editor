# Document Management with Generative AI

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)

A Streamlit-based document editor powered by generative AI, featuring rich text editing, AI-assisted feedback, and automated content refinement.

## Overview

This application combines traditional document editing with modern AI capabilities to create a powerful writing assistant. It offers:

- 📝 Rich text editing with Quill editor
- 🤖 AI-powered feedback generation
- 📑 Automated document summarization
- 📖 Distraction-free read mode
- 💾 Persistent state management

## Features

### Document Management
- Upload `.md`/`.txt` files
- Create new documents
- Download edited documents
- Copy to clipboard functionality
- Cache management
- Auto-save capabilities

### AI-Powered Features
- General document feedback
- Context-specific feedback for selected text
- Custom feedback management
- Content revision based on feedback
- Section-level summarization
- Summary-driven content updates

### User Interface
- Rich text editor (Quill)
- Distraction-free read mode
- Split-view editing
- Section management
- Mobile-responsive design

## Installation

1. Clone the repository:
~~~
git clone https://github.com/clchinkc/streamlit-editor.git
cd streamlit-editor
~~~

2. Install dependencies:
~~~
pip install -r requirements.txt
~~~

3. Configure environment variables (copy `.env.example` to `.env`):
~~~
# Required for AI features (choose one):
GITHUB_TOKEN=your_github_token
# OR
GEMINI_API_KEY=your_gemini_key
~~~

4. Launch the application:
~~~
streamlit run streamlit_editor.py
~~~

## Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `streamlit-quill` | Rich text editor integration |
| `dspy` | AI model management |
| `python-dotenv` | Environment configuration |
| `pyperclip` | Clipboard operations |
| `langchain-text-splitters` | Document splitting |

## Usage

### Basic Workflow

1. **Start Document**
   - Upload existing file or create new document
   - Use rich text editor for content creation

2. **AI Assistance**
   - Generate general document feedback
   - Get feedback for selected text
   - Add custom feedback items
   - Apply selected feedback to document

3. **Document Organization**
   - View and edit section summaries
   - Regenerate summaries as needed
   - Add/remove document sections
   - Update content from summaries

4. **Reading & Export**
   - Toggle read mode for distraction-free viewing
   - Download document
   - Copy content to clipboard

### Advanced Features

#### Summary Management
- Automatic section detection
- AI-powered summary generation
- Section-by-section editing
- Content regeneration from summaries

#### Feedback System
- Context-aware feedback generation
- Custom feedback integration
- Batch feedback application
- Reference text tracking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

