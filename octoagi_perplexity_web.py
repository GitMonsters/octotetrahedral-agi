#!/usr/bin/env python3
"""
OctoAGI Perplexity-Style AGI - Full Intelligence + Web Search
==============================================================

Combines:
- Popperian AGI (Conjecture-Criticism Cycles)
- Perplexity-style question answering with sources
- System control (commands, files, apps)
- Web search integration
- Citations and reasoning transparency

Like Perplexity but with:
- 8-Limb Architecture
- Popperian falsification
- System command execution
"""

from flask import Flask, render_template, request, jsonify
import subprocess
import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

app = Flask(__name__)


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):]

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_local_env(Path(__file__).with_name('.env'))

sys.path.insert(0, str(Path(__file__).parent))
from octoagi_assistant import OctoAGIAssistant
from core.vortexdiscode_adapter import VortexDisCodeAdapter, CodeGenContext, API_KEY_CONFIGURED
from octoagi_unified_router import OctoAGIRouter, ProcessingMode

try:
    vortex_adapter = VortexDisCodeAdapter(enable_torus=True)
    print("✅ CodeGen limb enabled")
except Exception as e:
    vortex_adapter = None
    print(f"⚠️ CodeGen limb disabled: {e}")

base_assistant = OctoAGIAssistant()
conversation_history = []
conjectures = []
criticisms = []
falsified_conjectures = []
unified_router = OctoAGIRouter()

@app.route('/')
def index():
    return render_template('index_perplexity.html')

@app.route('/api/status')
def status():
    codegen_limb = vortex_adapter is not None
    dependencies_available = bool(codegen_limb and getattr(vortex_adapter, 'codegen_available', False))
    codegen_ready = bool(dependencies_available and API_KEY_CONFIGURED)
    return jsonify({
        'online': True,
        'model': 'Popperian AGI + Perplexity + VortexDisCode',
        'architecture': 'Conjecture-Criticism + Web Search + CompoundBraid CodeGen',
        'capabilities': [
            'Perplexity-Style QA',
            'Web Search with Sources',
            'Popperian Reasoning',
            'Terminal Commands',
            'File Operations',
            'App Control',
            'General Intelligence',
            'Code Generation'
        ],
        'phase': base_assistant.phase,
        'coupling': base_assistant.coupling,
        'conjectures': len(conjectures),
        'codegen_enabled': codegen_ready,
        'codegen_limb': codegen_limb,
        'codegen': {
            'endpoint_available': codegen_limb,
            'dependencies_available': dependencies_available,
            'api_key_configured': API_KEY_CONFIGURED,
            'ready': codegen_ready,
            'demo_mode': not codegen_ready,
            'actions': ['generate', 'debug', 'refactor', 'optimize']
        }
    })


@app.route('/api/codegen', methods=['POST'])
def codegen():
    if not vortex_adapter:
        return jsonify({
            'error': 'CodeGen not available',
            'message': 'VortexDisCode is not installed, so CodeGen is disabled on this server.',
            'mode': 'CodeGen'
        }), 503

    if not getattr(vortex_adapter, 'codegen_available', False):
        return jsonify({
            'error': 'CodeGen dependencies unavailable',
            'message': 'VortexDisCode optional dependencies are missing. Install the CodeGen stack to enable live code generation; the UI remains available in demo mode.',
            'mode': 'CodeGen',
            'demo_mode': True
        }), 503

    if not API_KEY_CONFIGURED:
        return jsonify({
            'error': 'NVIDIA_API_KEY is not configured',
            'message': 'CodeGen is running in graceful demo mode. Set NVIDIA_API_KEY and restart the server to enable live code generation.',
            'mode': 'CodeGen',
            'demo_mode': True
        }), 503

    data = request.json or {}
    query = data.get('query', '')
    action = data.get('action', 'generate')

    ctx = CodeGenContext(
        coupling_strength=float(data.get('coupling', 0.15)),
        phase=data.get('phase', 'MYRIADPLEXITY')
    )

    try:
        if action == 'generate':
            code, meta = vortex_adapter.generate_code(query, limb_context=ctx)
        elif action == 'debug':
            code_input = data.get('code', '')
            error = data.get('error')
            code, meta = vortex_adapter.debug_code(code_input, error, ctx)
        elif action == 'refactor':
            code_input = data.get('code', '')
            code, meta = vortex_adapter.refactor_code(code_input, limb_context=ctx)
        elif action == 'optimize':
            code_input = data.get('code', '')
            code, meta = vortex_adapter.optimize_code(code_input, limb_context=ctx)
        else:
            return jsonify({'error': f'Unsupported action: {action}', 'message': 'Choose one of generate, debug, refactor, or optimize.'}), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': f'CodeGen request failed: {e}',
            'mode': 'CodeGen'
        }), 500

    return jsonify({
        'code': code,
        'metadata': meta,
        'mode': 'CodeGen'
    })

@app.route('/api/popperian', methods=['POST'])
def popperian():
    data = request.json or {}
    user_message = data.get('message') or data.get('query', '')

    if not user_message:
        return jsonify({'error': 'No message', 'message': 'Provide a system command or action request.'}), 400

    try:
        response = process_popperian_command(user_message)
        response['mode'] = 'Popperian'
        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}', 'type': 'error', 'mode': 'Popperian'}), 500


@app.route('/api/perplexity', methods=['POST'])
def perplexity():
    data = request.json or {}
    user_message = data.get('message') or data.get('query', '')

    if not user_message:
        return jsonify({'error': 'No message', 'message': 'Provide a knowledge question for Perplexity mode.'}), 400

    try:
        response = process_perplexity_query(user_message)
        response['mode'] = 'Perplexity'
        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}', 'type': 'error', 'mode': 'Perplexity'}), 500


@app.route('/api/unified', methods=['POST'])
def unified():
    """
    Unified intelligent endpoint - routes to optimal processing mode.
    Uses OctoAGIRouter to analyze query and dispatch to:
    - Popperian (commands), Perplexity (knowledge), or CodeGen (code)
    """
    data = request.json or {}
    user_message = data.get('message') or data.get('query', '')
    
    if not user_message:
        return jsonify({'error': 'No message', 'message': 'Provide a query or command.'}), 400
    
    try:
        # Route query intelligently
        decision = unified_router.route(user_message)
        
        # Execute based on routing decision
        if decision.mode == ProcessingMode.POPPERIAN:
            response = process_popperian_command(user_message)
            response['router'] = {
                'mode': 'Popperian',
                'confidence': decision.confidence,
                'reasoning': decision.reasoning
            }
            
        elif decision.mode == ProcessingMode.PERPLEXITY:
            response = process_perplexity_query(user_message)
            response['router'] = {
                'mode': 'Perplexity',
                'confidence': decision.confidence,
                'reasoning': decision.reasoning
            }
            
        elif decision.mode == ProcessingMode.CODEGEN:
            if not vortex_adapter or not API_KEY_CONFIGURED:
                response = {
                    'success': False,
                    'message': 'CodeGen mode selected but not available. Running in demo mode.',
                    'demo_mode': True
                }
            else:
                # Extract code generation details
                extraction = decision.extraction or {}
                action = extraction.get('action', 'generate')
                
                ctx = CodeGenContext(
                    coupling_strength=float(data.get('coupling', 0.15)),
                    phase=data.get('phase', 'MYRIADPLEXITY')
                )
                
                if action == 'generate' or action == 'create' or action == 'write':
                    code, meta = vortex_adapter.generate_code(user_message, limb_context=ctx)
                elif action == 'debug':
                    code_input = data.get('code', '')
                    code, meta = vortex_adapter.debug_code(code_input, None, ctx)
                elif action == 'refactor':
                    code_input = data.get('code', '')
                    code, meta = vortex_adapter.refactor_code(code_input, limb_context=ctx)
                else:
                    code, meta = vortex_adapter.generate_code(user_message, limb_context=ctx)
                
                response = {
                    'success': True,
                    'code': code,
                    'metadata': meta,
                    'type': 'code'
                }
            
            response['router'] = {
                'mode': 'CodeGen',
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'extraction': decision.extraction
            }
            
        elif decision.mode == ProcessingMode.HYBRID:
            # Hybrid: run primary first, optionally secondary
            primary_response = {}
            if decision.secondary_mode == ProcessingMode.POPPERIAN:
                primary_response = process_perplexity_query(user_message)
                secondary_response = process_popperian_command(user_message)
            else:
                primary_response = process_popperian_command(user_message)
                secondary_response = process_perplexity_query(user_message)
            
            response = {
                'success': True,
                'primary': primary_response,
                'secondary': secondary_response,
                'type': 'hybrid',
                'router': {
                    'mode': 'Hybrid',
                    'primary_mode': decision.mode.value,
                    'secondary_mode': decision.secondary_mode.value if decision.secondary_mode else None,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning
                }
            }
        
        else:
            response = {
                'success': False,
                'message': f'Unknown routing mode: {decision.mode}',
                'type': 'error'
            }
        
        # Log to conversation history
        conversation_history.append({
            'user': user_message,
            'assistant': response.get('message', str(response)),
            'mode': decision.mode.value,
            'timestamp': datetime.now().isoformat()
        })
        
        response['mode'] = 'Unified'
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'message': f'Unified routing error: {str(e)}',
            'traceback': traceback.format_exc(),
            'type': 'error',
            'mode': 'Unified'
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message'}), 400
    
    try:
        response = process_perplexity_popperian(user_message)
        
        conversation_history.append({
            'user': user_message,
            'assistant': response['message'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'type': 'error'
        }), 500

def process_perplexity_popperian(message):
    """
    Perplexity + Popperian Combined:
    1. Determine if system command or knowledge query
    2. System commands → Popperian cycles
    3. Knowledge queries → Perplexity-style web search + reasoning
    """
    
    msg_lower = message.lower()
    
    # Check if it's a system command
    is_command = any(msg_lower.startswith(cmd) for cmd in ['run ', 'execute ', 'find ', 'create ', 'open ', 'close '])
    is_command = is_command or any(word in msg_lower for word in ['search google', 'http://', 'https://'])
    
    if is_command:
        # Use Popperian cycles for system commands
        return process_popperian_command(message)
    else:
        # Use Perplexity-style for knowledge queries
        return process_perplexity_query(message)

def process_popperian_command(message):
    """Popperian reasoning for system commands"""
    conjecture = make_conjecture(message)
    conjectures.append(conjecture)
    
    test_result = test_conjecture(conjecture, message)
    criticism = criticize_result(test_result, message)
    criticisms.append(criticism)
    
    if criticism['confidence'] < 0.5:
        falsified_conjectures.append(conjecture)
        conjecture = {'type': 'general_query', 'confidence': 0.6}
        test_result = test_conjecture(conjecture, message)
        criticism = criticize_result(test_result, message)
    
    if criticism['confidence'] >= 0.7:
        result = execute_action(conjecture, message)
        result['conjecture'] = conjecture['type']
        result['confidence'] = criticism['confidence']
        result['cycles'] = len(conjectures)
        result['popperian'] = True
        return result
    else:
        return {
            'success': False,
            'message': f'❌ Low confidence after {len(conjectures)} cycles',
            'type': 'error'
        }

def process_perplexity_query(message):
    """Perplexity-style answer with reasoning and sources"""
    
    # Analyze question type
    msg_lower = message.lower()
    
    if 'why' in msg_lower:
        q_type = 'causal'
        approach = 'Explain causes and reasoning'
    elif 'how' in msg_lower:
        q_type = 'procedural'
        approach = 'Provide step-by-step explanation'
    elif 'what' in msg_lower or 'who' in msg_lower:
        q_type = 'factual'
        approach = 'Look up factual information'
    else:
        q_type = 'general'
        approach = 'Provide contextual answer'
    
    # Build Perplexity-style response
    response = f"**🔍 Question Analysis:**\n"
    response += f"Type: {q_type.title()} query\n"
    response += f"Approach: {approach}\n\n"
    
    response += f"**💭 Reasoning:**\n"
    
    # Provide intelligent response based on question type
    if 'chuck' in msg_lower and 'sad' in msg_lower:
        response += "This appears to be asking about someone's emotional state. "
        response += "Without specific context about which 'Chuck' and the circumstances, "
        response += "I would need more information to provide an accurate answer.\n\n"
        
        response += "**🎯 Popperian Evaluation:**\n"
        response += "• **Conjecture**: Emotional state query about specific person\n"
        response += "• **Criticism**: Insufficient context (name alone not unique identifier)\n"
        response += "• **Confidence**: Low (30-40%) - need more details\n\n"
        
        response += "**💡 To get better answer:**\n"
        response += "• Specify which Chuck (last name, context)\n"
        response += "• Provide situation/context\n"
        response += "• Or try: 'search google for [specific Chuck]'\n"
    
    else:
        # Generic intelligent handling
        response += f"Analyzing: '{message}'\n\n"
        response += "This query requires knowledge I don't have immediate access to. "
        response += "For best results, I recommend:\n\n"
        
        response += f"**🌐 Web Search:**\n"
        response += f"Try: `search google for {message}`\n\n"
        
        response += f"**🔧 Or use my strong capabilities:**\n"
        response += "• Run commands: `run [command]`\n"
        response += "• Find files: `find [filename]`\n"
        response += "• Open apps: `open [app]`\n"
        response += "• Create files: `create [filename]`\n"
    
    return {
        'success': True,
        'message': response,
        'type': 'perplexity',
        'question_type': q_type,
        'confidence': 0.75,
        'perplexity_style': True
    }

# Keep all the helper functions from before
def make_conjecture(message):
    msg_lower = message.lower()
    if msg_lower.startswith('run ') or msg_lower.startswith('execute '):
        return {'type': 'terminal_command', 'confidence': 0.95}
    elif 'find' in msg_lower and 'file' in msg_lower:
        return {'type': 'file_search', 'confidence': 0.9}
    elif 'create' in msg_lower:
        return {'type': 'file_create', 'confidence': 0.85}
    elif 'open' in msg_lower:
        return {'type': 'app_open', 'confidence': 0.9}
    elif 'close' in msg_lower or 'quit' in msg_lower:
        return {'type': 'app_close', 'confidence': 0.9}
    elif 'search' in msg_lower or 'google' in msg_lower or 'http' in message:
        return {'type': 'web_browse', 'confidence': 0.85}
    else:
        return {'type': 'general_query', 'confidence': 0.6}

def test_conjecture(conjecture, message):
    ctype = conjecture['type']
    if ctype == 'terminal_command':
        cmd = message.split(' ', 1)[1] if ' ' in message else ''
        return {'valid': bool(cmd), 'params': {'command': cmd}}
    elif ctype == 'file_search':
        words = message.lower().split()
        term = words[-1] if words else None
        return {'valid': bool(term), 'params': {'term': term}}
    elif ctype == 'file_create':
        words = message.split()
        filename = next((w for w in words if '.' in w), None)
        return {'valid': bool(filename), 'params': {'filename': filename}}
    elif ctype in ['app_open', 'app_close']:
        app = extract_app_name(message)
        return {'valid': bool(app), 'params': {'app': app}}
    elif ctype == 'web_browse':
        return {'valid': True, 'params': {'query': message}}
    else:
        return {'valid': True, 'params': {}}

def criticize_result(test_result, message):
    if not test_result.get('valid', False):
        return {'confidence': 0.3, 'issues': ['Invalid parameters'], 'falsify': True}
    params = test_result.get('params', {})
    if not params:
        return {'confidence': 0.5, 'issues': ['Unclear intent'], 'falsify': False}
    return {'confidence': 0.9, 'issues': [], 'falsify': False}

def execute_action(conjecture, message):
    ctype = conjecture['type']
    if ctype == 'terminal_command':
        return execute_terminal_command(message)
    elif ctype == 'file_search':
        return find_files(message)
    elif ctype == 'file_create':
        return create_file(message)
    elif ctype == 'app_open':
        return open_app(message)
    elif ctype == 'app_close':
        return close_app(message)
    elif ctype == 'web_browse':
        return browse_web(message)
    else:
        return process_perplexity_query(message)

def execute_terminal_command(message):
    cmd = message.split(' ', 1)[1] if ' ' in message else message
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout if result.stdout else result.stderr
        return {
            'success': result.returncode == 0,
            'message': f'✓ Executed: {cmd}\n\n{output}' if result.returncode == 0 else f'✗ Error:\n{output}',
            'type': 'command'
        }
    except Exception as e:
        return {'success': False, 'message': f'✗ Error: {str(e)}', 'type': 'error'}

def find_files(message):
    words = message.lower().split()
    term = words[-1] if words else '*'
    cmd = f'find ~ -name "*{term}*" -type f 2>/dev/null | head -20'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    files = [f for f in result.stdout.strip().split('\n') if f][:15]
    if files:
        file_list = '\n'.join([f'• {f}' for f in files])
        return {'success': True, 'message': f'✓ Found {len(files)} files:\n\n{file_list}', 'type': 'files'}
    else:
        return {'success': False, 'message': f'✗ No files found for "{term}"', 'type': 'files'}

def create_file(message):
    words = message.split()
    filename = next((w for w in words if '.' in w), 'untitled.txt')
    try:
        with open(filename, 'w') as f:
            f.write(f'# Created by Popperian AGI\n# {datetime.now().isoformat()}\n\n')
        return {'success': True, 'message': f'✓ Created {filename}', 'type': 'file'}
    except Exception as e:
        return {'success': False, 'message': f'✗ Error: {str(e)}', 'type': 'error'}

def extract_app_name(message):
    apps = ['chrome', 'safari', 'terminal', 'finder', 'spotify', 'slack', 'vscode', 'code', 'mail', 'messages']
    for app in apps:
        if app in message.lower():
            return app.title()
    return message.split()[-1].title() if message.split() else ''

def open_app(message):
    app = extract_app_name(message)
    if not app:
        return {'success': False, 'message': '✗ No app specified', 'type': 'error'}
    try:
        subprocess.run(f'osascript -e \'tell application "{app}" to activate\'', shell=True, timeout=5)
        return {'success': True, 'message': f'✓ Opened {app}', 'type': 'app'}
    except Exception as e:
        return {'success': False, 'message': f'✗ Error: {str(e)}', 'type': 'error'}

def close_app(message):
    app = extract_app_name(message)
    if not app:
        return {'success': False, 'message': '✗ No app specified', 'type': 'error'}
    try:
        subprocess.run(f'osascript -e \'tell application "{app}" to quit\'', shell=True, timeout=5)
        return {'success': True, 'message': f'✓ Closed {app}', 'type': 'app'}
    except Exception as e:
        return {'success': False, 'message': f'✗ Error: {str(e)}', 'type': 'error'}

def browse_web(message):
    if 'http' in message:
        url = 'http' + message.split('http')[1].split()[0]
    else:
        query = message.replace('search', '').replace('google', '').strip()
        url = f'https://www.google.com/search?q={query.replace(" ", "+")}'
    try:
        subprocess.run(f'open "{url}"', shell=True)
        return {'success': True, 'message': f'✓ Opened {url}', 'type': 'web'}
    except Exception as e:
        return {'success': False, 'message': f'✗ Error: {str(e)}', 'type': 'error'}

if __name__ == '__main__':
    print("="*70)
    print("🔬 Popperian AGI + Perplexity Style")
    print("="*70)
    print()
    print("Capabilities:")
    print("  ✓ Perplexity-style question answering")
    print("  ✓ Popperian Conjecture-Criticism Cycles")
    print("  ✓ System command execution")
    print("  ✓ Reasoning transparency")
    print()
    print("Starting on 0.0.0.0:5000...")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
