#!/usr/bin/env python3
"""
OCTO Live Server
Real-time WebSocket server for OCTO RNA Editing Layer visualization.

Provides:
- REST API for analysis, training, status
- WebSocket for real-time streaming
- Multi-strategy text embedding
- Serves the live dashboard
"""

import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import deque

import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.httpserver

# Add OCTO path
sys.path.insert(0, '/home/worm/octotetrahedral-agi')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('octo-server')

# =============================================================================
# Text Embedder - Multi-Strategy
# =============================================================================

class TextEmbedder:
    """
    Multi-strategy text embedding for converting text to 256-dim vectors.
    
    Strategies (in order of preference):
    1. Semantic features (explicit)
    2. Character n-gram hashing
    3. TF-IDF projection (if sklearn available)
    """
    
    # Common code keywords for detection
    CODE_KEYWORDS = {
        'def', 'class', 'function', 'return', 'import', 'from', 'const', 'let', 'var',
        'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'catch', 'throw',
        'public', 'private', 'static', 'void', 'int', 'string', 'bool', 'float',
        'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'JOIN', 'ORDER'
    }
    
    # Emotional words (simplified lexicon)
    POSITIVE_WORDS = {
        'happy', 'joy', 'love', 'great', 'wonderful', 'amazing', 'excellent',
        'good', 'best', 'fantastic', 'beautiful', 'awesome', 'excited', 'glad'
    }
    NEGATIVE_WORDS = {
        'sad', 'angry', 'hate', 'terrible', 'awful', 'bad', 'worst', 'horrible',
        'depressed', 'anxious', 'stressed', 'frustrated', 'upset', 'worried'
    }
    
    # Technical terms
    TECHNICAL_TERMS = {
        'algorithm', 'function', 'variable', 'parameter', 'array', 'object',
        'database', 'server', 'client', 'api', 'endpoint', 'request', 'response',
        'memory', 'cpu', 'gpu', 'thread', 'process', 'network', 'protocol',
        'integral', 'derivative', 'equation', 'theorem', 'proof', 'hypothesis'
    }
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.semantic_dim = 32  # Explicit semantic features
        self.hash_dim = dim - self.semantic_dim  # Remaining for hashing
        
        # Try to initialize TF-IDF
        self.tfidf = None
        self.svd = None
        self._init_tfidf()
        
        logger.info(f"TextEmbedder initialized (dim={dim}, tfidf={'yes' if self.tfidf else 'no'})")
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer with common vocabulary."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            # Pre-fit on diverse sample texts
            sample_texts = [
                "def function return value parameter argument",
                "hello how are you doing today friend",
                "calculate integral derivative equation math",
                "write story creative poem imagination",
                "feeling happy sad angry emotional",
                "SELECT FROM WHERE JOIN database query",
                "error bug debug fix issue problem",
                "explain why how what when where",
                "algorithm complexity time space memory",
                "user interface design button click event",
            ]
            
            self.tfidf = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_matrix = self.tfidf.fit_transform(sample_texts)
            
            # SVD to reduce dimensions
            n_components = min(50, tfidf_matrix.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=n_components)
            self.svd.fit(tfidf_matrix)
            
        except ImportError:
            logger.warning("sklearn not available, using hash-only embedding")
            self.tfidf = None
            self.svd = None
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """Extract explicit semantic features from text."""
        features = np.zeros(self.semantic_dim)
        
        text_lower = text.lower()
        words = text_lower.split()
        word_set = set(words)
        
        # Length features (0-3)
        features[0] = min(len(text) / 500, 1.0)  # Normalized length
        features[1] = min(len(words) / 100, 1.0)  # Word count
        features[2] = np.mean([len(w) for w in words]) / 10 if words else 0  # Avg word length
        features[3] = len(set(words)) / max(len(words), 1)  # Vocabulary diversity
        
        # Punctuation features (4-7)
        features[4] = text.count('?') / max(len(text), 1) * 100  # Questions
        features[5] = text.count('!') / max(len(text), 1) * 100  # Exclamations
        features[6] = (text.count('(') + text.count('{') + text.count('[')) / max(len(text), 1) * 50  # Brackets
        features[7] = text.count(';') / max(len(text), 1) * 50  # Semicolons (code indicator)
        
        # Code detection (8-11)
        code_word_count = len(word_set & self.CODE_KEYWORDS)
        features[8] = min(code_word_count / 5, 1.0)  # Code keyword density
        features[9] = 1.0 if any(c in text for c in ['=>', '->', '==', '!=', '&&', '||']) else 0  # Operators
        features[10] = 1.0 if any(text.strip().startswith(kw) for kw in ['def ', 'class ', 'function ', 'SELECT ']) else 0
        features[11] = text.count('_') / max(len(text), 1) * 20  # Underscores (variable names)
        
        # Emotional features (12-15)
        positive_count = len(word_set & self.POSITIVE_WORDS)
        negative_count = len(word_set & self.NEGATIVE_WORDS)
        features[12] = min(positive_count / 3, 1.0)  # Positive sentiment
        features[13] = min(negative_count / 3, 1.0)  # Negative sentiment
        features[14] = abs(positive_count - negative_count) / max(positive_count + negative_count, 1)  # Polarity
        features[15] = 1.0 if any(w in text_lower for w in ['feel', 'feeling', 'felt']) else 0  # Emotional framing
        
        # Technical features (16-19)
        tech_count = len(word_set & self.TECHNICAL_TERMS)
        features[16] = min(tech_count / 5, 1.0)  # Technical term density
        features[17] = 1.0 if any(w in text_lower for w in ['calculate', 'compute', 'solve', 'prove']) else 0
        features[18] = 1.0 if any(w in text_lower for w in ['integral', 'derivative', 'equation', 'theorem']) else 0
        features[19] = sum(1 for c in text if c.isupper()) / max(len(text), 1)  # Uppercase ratio
        
        # Question/instruction features (20-23)
        features[20] = 1.0 if text.strip().endswith('?') else 0  # Is question
        features[21] = 1.0 if any(text_lower.startswith(w) for w in ['what', 'why', 'how', 'when', 'where', 'who']) else 0
        features[22] = 1.0 if any(text_lower.startswith(w) for w in ['write', 'create', 'make', 'build', 'generate']) else 0
        features[23] = 1.0 if any(text_lower.startswith(w) for w in ['explain', 'describe', 'tell', 'help']) else 0
        
        # Creative features (24-27)
        features[24] = 1.0 if any(w in text_lower for w in ['imagine', 'creative', 'story', 'poem', 'dream']) else 0
        features[25] = 1.0 if any(w in text_lower for w in ['haiku', 'sonnet', 'verse', 'rhyme']) else 0
        features[26] = 1.0 if any(w in text_lower for w in ['once upon', 'in a world', 'there was']) else 0
        features[27] = len([w for w in words if len(w) > 10]) / max(len(words), 1)  # Long word ratio
        
        # Formality features (28-31)
        features[28] = 1.0 if any(w in text_lower for w in ['please', 'thank', 'would', 'could', 'kindly']) else 0
        features[29] = 1.0 if any(w in text_lower for w in ['hey', 'hi', 'yo', 'sup', 'gonna', 'wanna']) else 0
        features[30] = text.count(',') / max(len(text), 1) * 50  # Comma usage (complexity)
        features[31] = 1.0 if text[0].isupper() and text.endswith('.') else 0  # Proper sentence
        
        return features
    
    def _hash_ngrams(self, text: str) -> np.ndarray:
        """Generate embedding from character n-gram hashing."""
        embedding = np.zeros(self.hash_dim)
        
        # Extract n-grams
        text_lower = text.lower()
        ngrams = []
        for n in [2, 3, 4, 5]:
            for i in range(len(text_lower) - n + 1):
                ngrams.append(text_lower[i:i+n])
        
        # Also add word unigrams and bigrams
        words = text_lower.split()
        ngrams.extend(words)
        for i in range(len(words) - 1):
            ngrams.append(f"{words[i]} {words[i+1]}")
        
        # Hash each n-gram to a position
        for gram in ngrams:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            pos = h % self.hash_dim
            # Use different bits for value
            val = ((h >> 8) % 1000) / 500.0 - 1.0  # Value in [-1, 1]
            embedding[pos] += val
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _tfidf_embed(self, text: str) -> np.ndarray:
        """Generate embedding using TF-IDF + SVD."""
        if self.tfidf is None or self.svd is None:
            return np.zeros(50)
        
        try:
            tfidf_vec = self.tfidf.transform([text])
            reduced = self.svd.transform(tfidf_vec)
            return reduced[0]
        except:
            return np.zeros(50)
    
    def embed(self, text: str) -> List[float]:
        """
        Convert text to 256-dimensional embedding.
        
        Combines:
        - Semantic features (32 dims)
        - Hash-based n-grams (remaining dims)
        - Optional TF-IDF blending
        """
        if not text.strip():
            return [0.0] * self.dim
        
        # Extract semantic features
        semantic = self._extract_semantic_features(text)
        
        # Hash-based embedding
        hashed = self._hash_ngrams(text)
        
        # Combine
        embedding = np.concatenate([semantic, hashed])
        
        # Optionally blend with TF-IDF
        if self.tfidf is not None:
            tfidf_emb = self._tfidf_embed(text)
            # Blend into hash portion
            blend_size = min(len(tfidf_emb), self.hash_dim)
            embedding[self.semantic_dim:self.semantic_dim + blend_size] += tfidf_emb[:blend_size] * 0.3
        
        # Final normalization (preserve semantic features scale)
        hash_part = embedding[self.semantic_dim:]
        norm = np.linalg.norm(hash_part)
        if norm > 0:
            embedding[self.semantic_dim:] = hash_part / norm
        
        return embedding.tolist()
    
    def get_input_type(self, text: str) -> Dict[str, Any]:
        """Detect input type from text."""
        features = self._extract_semantic_features(text)
        
        # Determine type based on features
        scores = {
            'code': features[8] + features[9] + features[10] + features[11],
            'math': features[17] + features[18] + features[16],
            'creative': features[24] + features[25] + features[26],
            'emotional': features[12] + features[13] + features[15],
            'question': features[20] + features[21],
            'instruction': features[22] + features[23],
        }
        
        # Find max
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        if max_score < 0.3:
            return {'type': 'conversational', 'confidence': 0.5}
        
        return {'type': max_type, 'confidence': min(max_score / 2, 1.0)}


# =============================================================================
# Analysis History
# =============================================================================

class AnalysisHistory:
    """Track recent analyses."""
    
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self.items = deque(maxlen=max_items)
        self.total_count = 0
        self.total_latency = 0.0
    
    def add(self, item: Dict[str, Any]):
        """Add an analysis result to history."""
        item['id'] = self.total_count
        item['timestamp'] = datetime.now().isoformat()
        self.items.appendleft(item)
        self.total_count += 1
        self.total_latency += item.get('latency_ms', 0)
    
    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get recent items."""
        return list(self.items)[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get history statistics."""
        if self.total_count == 0:
            return {'count': 0, 'avg_latency_ms': 0}
        return {
            'count': self.total_count,
            'avg_latency_ms': self.total_latency / self.total_count
        }


# =============================================================================
# Global State
# =============================================================================

class OCTOState:
    """Global server state."""
    
    def __init__(self):
        self.embedder = TextEmbedder(dim=256)
        self.bridge = None
        self.history = AnalysisHistory()
        self.websockets = set()
        self.start_time = time.time()
        self.weights_loaded = False
        self.weights_path = '/home/worm/octotetrahedral-agi/rna_weights.pt'
        
        # Initialize bridge
        self._init_bridge()
    
    def _init_bridge(self):
        """Initialize the OCTO bridge."""
        try:
            import rustyworm_bridge as bridge
            self.bridge = bridge.get_bridge(256)
            
            # Try to load weights
            if os.path.exists(self.weights_path):
                if bridge.load_weights(self.weights_path):
                    self.weights_loaded = True
                    logger.info(f"Loaded weights from {self.weights_path}")
                else:
                    logger.warning(f"Failed to load weights from {self.weights_path}")
            else:
                logger.info("No pre-trained weights found, using random initialization")
                
        except Exception as e:
            logger.error(f"Failed to initialize OCTO bridge: {e}")
            self.bridge = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text through OCTO."""
        start = time.time()
        
        # Get embedding
        embedding = self.embedder.embed(text)
        input_type = self.embedder.get_input_type(text)
        
        # Analyze through OCTO
        if self.bridge:
            try:
                result = self.bridge.analyze(embedding)
                routing = self.bridge.get_routing_decision(embedding)
                
                # Determine route
                use_system1 = routing['route'] == 'system1'
                primary_pathway = routing['primary_pathway']
                pathway_name = self.bridge.get_pathway_name(primary_pathway)
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                result = self._fallback_analysis()
                use_system1 = False
                primary_pathway = 1
                pathway_name = 'reasoning'
        else:
            result = self._fallback_analysis()
            use_system1 = False
            primary_pathway = 1
            pathway_name = 'reasoning'
        
        latency = (time.time() - start) * 1000
        
        analysis = {
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'input_type': input_type,
            'confidence': result['confidence'],
            'temperature': result['temperature'],
            'head_gates': result['head_gates'],
            'pathway_weights': result['pathway_weights'],
            'route': 'system1' if use_system1 else 'system2',
            'primary_pathway': primary_pathway,
            'pathway_name': pathway_name,
            'latency_ms': round(latency, 2)
        }
        
        # Add to history
        self.history.add(analysis.copy())
        
        return analysis
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback when bridge is unavailable."""
        return {
            'confidence': 0.5,
            'temperature': 2.0,
            'head_gates': [0.5] * 8,
            'pathway_weights': [0.33, 0.34, 0.33]
        }
    
    def train(self, text: str, target_pathway: int, target_confidence: float, epochs: int = 10) -> Dict[str, Any]:
        """Train the model on an example."""
        if not self.bridge:
            return {'error': 'Bridge not initialized'}
        
        try:
            embedding = self.embedder.embed(text)
            
            if epochs == 1:
                result = self.bridge.train_step(
                    embedding, target_pathway, target_confidence, learning_rate=0.01
                )
            else:
                result = self.bridge.train_batch(
                    [embedding], [target_pathway], [target_confidence],
                    epochs=epochs, learning_rate=0.01
                )
            
            return {
                'success': True,
                'epochs': epochs,
                **result
            }
        except Exception as e:
            return {'error': str(e)}
    
    def save_weights(self) -> Dict[str, Any]:
        """Save model weights."""
        if not self.bridge:
            return {'error': 'Bridge not initialized'}
        
        try:
            path = self.bridge.save_weights(self.weights_path)
            self.weights_loaded = True
            return {'success': True, 'path': path}
        except Exception as e:
            return {'error': str(e)}
    
    def load_weights(self) -> Dict[str, Any]:
        """Load model weights."""
        if not self.bridge:
            return {'error': 'Bridge not initialized'}
        
        try:
            if self.bridge.load_weights(self.weights_path):
                self.weights_loaded = True
                return {'success': True, 'path': self.weights_path}
            return {'error': 'Failed to load weights'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        uptime = time.time() - self.start_time
        
        return {
            'status': 'online',
            'bridge_active': self.bridge is not None,
            'weights_loaded': self.weights_loaded,
            'weights_path': self.weights_path,
            'device': 'cpu',
            'embedding_dim': 256,
            'embedder_type': 'hash+semantic' + ('+tfidf' if self.embedder.tfidf else ''),
            'uptime_seconds': round(uptime, 1),
            'uptime_formatted': f"{int(uptime // 60)}m {int(uptime % 60)}s",
            'websocket_clients': len(self.websockets),
            'history_stats': self.history.get_stats()
        }
    
    def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients."""
        msg_str = json.dumps(message)
        for ws in list(self.websockets):
            try:
                ws.write_message(msg_str)
            except:
                self.websockets.discard(ws)


# Global state instance
state = OCTOState()


# =============================================================================
# HTTP Handlers
# =============================================================================

class MainHandler(tornado.web.RequestHandler):
    """Serve the main dashboard."""
    
    def get(self):
        self.redirect('/dashboard')


class DashboardHandler(tornado.web.RequestHandler):
    """Serve the live dashboard HTML."""
    
    def get(self):
        dashboard_path = os.path.join(os.path.dirname(__file__), 'octo_live.html')
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r') as f:
                self.write(f.read())
        else:
            self.write("Dashboard not found. Please create octo_live.html")


class AnalyzeHandler(tornado.web.RequestHandler):
    """Handle analysis requests."""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
    
    def options(self):
        self.set_status(204)
        self.finish()
    
    def post(self):
        try:
            data = json.loads(self.request.body)
            text = data.get('text', '')
            
            if not text.strip():
                self.set_status(400)
                self.write({'error': 'No text provided'})
                return
            
            result = state.analyze(text)
            
            # Broadcast to WebSocket clients
            state.broadcast({'type': 'analysis', 'data': result})
            
            self.write(result)
            
        except Exception as e:
            self.set_status(500)
            self.write({'error': str(e)})


class TrainHandler(tornado.web.RequestHandler):
    """Handle training requests."""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
    
    def options(self):
        self.set_status(204)
        self.finish()
    
    def post(self):
        try:
            data = json.loads(self.request.body)
            text = data.get('text', '')
            target_pathway = data.get('pathway', 1)
            target_confidence = data.get('confidence', 0.8)
            epochs = data.get('epochs', 10)
            
            if not text.strip():
                self.set_status(400)
                self.write({'error': 'No text provided'})
                return
            
            result = state.train(text, target_pathway, target_confidence, epochs)
            
            # Broadcast training result
            state.broadcast({'type': 'training', 'data': result})
            
            self.write(result)
            
        except Exception as e:
            self.set_status(500)
            self.write({'error': str(e)})


class StatusHandler(tornado.web.RequestHandler):
    """Handle status requests."""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
    
    def get(self):
        self.write(state.get_status())


class HistoryHandler(tornado.web.RequestHandler):
    """Handle history requests."""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
    
    def get(self):
        limit = int(self.get_argument('limit', 20))
        self.write({'history': state.history.get_recent(limit)})


class SaveWeightsHandler(tornado.web.RequestHandler):
    """Handle weight saving."""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
    
    def options(self):
        self.set_status(204)
        self.finish()
    
    def post(self):
        result = state.save_weights()
        self.write(result)


class LoadWeightsHandler(tornado.web.RequestHandler):
    """Handle weight loading."""
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
    
    def options(self):
        self.set_status(204)
        self.finish()
    
    def post(self):
        result = state.load_weights()
        self.write(result)


# =============================================================================
# WebSocket Handler
# =============================================================================

class OCTOWebSocket(tornado.websocket.WebSocketHandler):
    """WebSocket handler for real-time communication."""
    
    def check_origin(self, origin):
        return True  # Allow all origins for development
    
    def open(self):
        state.websockets.add(self)
        logger.info(f"WebSocket opened. Total clients: {len(state.websockets)}")
        
        # Send initial status
        self.write_message(json.dumps({
            'type': 'connected',
            'data': state.get_status()
        }))
    
    def on_message(self, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'analyze':
                text = data.get('text', '')
                if text.strip():
                    result = state.analyze(text)
                    self.write_message(json.dumps({
                        'type': 'analysis',
                        'data': result
                    }))
            
            elif action == 'train':
                text = data.get('text', '')
                pathway = data.get('pathway', 1)
                confidence = data.get('confidence', 0.8)
                epochs = data.get('epochs', 10)
                
                if text.strip():
                    result = state.train(text, pathway, confidence, epochs)
                    self.write_message(json.dumps({
                        'type': 'training',
                        'data': result
                    }))
            
            elif action == 'status':
                self.write_message(json.dumps({
                    'type': 'status',
                    'data': state.get_status()
                }))
            
            elif action == 'history':
                limit = data.get('limit', 20)
                self.write_message(json.dumps({
                    'type': 'history',
                    'data': state.history.get_recent(limit)
                }))
            
            elif action == 'save':
                result = state.save_weights()
                self.write_message(json.dumps({
                    'type': 'save',
                    'data': result
                }))
            
            elif action == 'load':
                result = state.load_weights()
                self.write_message(json.dumps({
                    'type': 'load',
                    'data': result
                }))
                
        except json.JSONDecodeError:
            self.write_message(json.dumps({
                'type': 'error',
                'data': {'error': 'Invalid JSON'}
            }))
        except Exception as e:
            self.write_message(json.dumps({
                'type': 'error',
                'data': {'error': str(e)}
            }))
    
    def on_close(self):
        state.websockets.discard(self)
        logger.info(f"WebSocket closed. Total clients: {len(state.websockets)}")


# =============================================================================
# Application Setup
# =============================================================================

def make_app():
    """Create the Tornado application."""
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/dashboard", DashboardHandler),
        (r"/api/analyze", AnalyzeHandler),
        (r"/api/train", TrainHandler),
        (r"/api/status", StatusHandler),
        (r"/api/history", HistoryHandler),
        (r"/api/save", SaveWeightsHandler),
        (r"/api/load", LoadWeightsHandler),
        (r"/ws", OCTOWebSocket),
    ], debug=True)


def main():
    """Main entry point."""
    port = int(os.environ.get('OCTO_PORT', 8888))
    
    app = make_app()
    app.listen(port)
    
    logger.info("=" * 60)
    logger.info("OCTO LIVE SERVER")
    logger.info("=" * 60)
    logger.info(f"Dashboard: http://localhost:{port}/dashboard")
    logger.info(f"WebSocket: ws://localhost:{port}/ws")
    logger.info(f"API:       http://localhost:{port}/api/")
    logger.info("=" * 60)
    logger.info("Endpoints:")
    logger.info("  POST /api/analyze - Analyze text")
    logger.info("  POST /api/train   - Train model")
    logger.info("  GET  /api/status  - Server status")
    logger.info("  GET  /api/history - Analysis history")
    logger.info("  POST /api/save    - Save weights")
    logger.info("  POST /api/load    - Load weights")
    logger.info("  WS   /ws          - WebSocket endpoint")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == "__main__":
    main()
