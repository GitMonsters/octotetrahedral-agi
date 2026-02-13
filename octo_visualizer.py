#!/usr/bin/env python3
"""
OCTO Consciousness Visualizer
Real-time ASCII visualization of RNA Editing Layer routing decisions.
"""

import sys
import time
import random
import math
import os

# Add path for rustyworm_bridge
sys.path.insert(0, '/home/worm/octotetrahedral-agi')

try:
    import rustyworm_bridge as bridge
except ImportError:
    print("Warning: rustyworm_bridge not available, using simulation mode")
    bridge = None

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    BLINK = '\033[5m'
    
    # Foreground
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def move_cursor(x, y):
    print(f'\033[{y};{x}H', end='')

def gradient_char(value, chars=" ░▒▓█"):
    """Convert 0-1 value to gradient character"""
    idx = int(value * (len(chars) - 1))
    return chars[min(idx, len(chars) - 1)]

def color_by_value(value):
    """Get color based on activation value"""
    if value > 0.8:
        return Colors.BRIGHT_GREEN
    elif value > 0.6:
        return Colors.GREEN
    elif value > 0.4:
        return Colors.YELLOW
    elif value > 0.2:
        return Colors.RED
    else:
        return Colors.DIM

def draw_brain():
    """Draw the OCTO brain ASCII art"""
    brain = f"""
{Colors.CYAN}                    ╔══════════════════════════════════════════════════════════════╗
                    ║  {Colors.BRIGHT_CYAN}█▀▀█ █▀▀ ▀▀█▀▀ █▀▀█   {Colors.BRIGHT_MAGENTA}█▀▀▄ █▀▀█ █▀▀█ ▀█▀ █▀▀▄{Colors.CYAN}  ║
                    ║  {Colors.BRIGHT_CYAN}█  █ █     █   █  █   {Colors.BRIGHT_MAGENTA}█▀▀▄ █▄▄▀ █▄▄█  █  █  █{Colors.CYAN}  ║
                    ║  {Colors.BRIGHT_CYAN}▀▀▀▀ ▀▀▀   ▀   ▀▀▀▀   {Colors.BRIGHT_MAGENTA}▀▀▀  ▀ ▀▀ ▀  ▀ ▀▀▀ ▀  ▀{Colors.CYAN}  ║
                    ╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
    """
    return brain

def draw_pathway_diagram(pathway_weights, primary_pathway, head_gates, confidence, temperature, route):
    """Draw the neural pathway visualization"""
    
    pw = pathway_weights
    hg = head_gates
    
    # Pathway colors based on activation
    p_colors = [color_by_value(w) for w in pw]
    h_colors = [color_by_value(g) for g in hg]
    
    # Route indicator
    if route == "System 1":
        route_color = Colors.BRIGHT_GREEN
        route_icon = "⚡"
    else:
        route_color = Colors.BRIGHT_BLUE
        route_icon = "🧠"
    
    # Confidence bar
    conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
    conf_color = Colors.BRIGHT_GREEN if confidence > 0.65 else Colors.YELLOW
    
    # Temperature gauge
    temp_normalized = min(temperature / 5.0, 1.0)
    temp_bar = "█" * int(temp_normalized * 20) + "░" * (20 - int(temp_normalized * 20))
    temp_color = Colors.BRIGHT_RED if temperature > 2.0 else Colors.CYAN
    
    # Pathway bars
    def pathway_bar(val, width=15):
        filled = int(val * width)
        return "█" * filled + "░" * (width - filled)
    
    # Head gate visualization
    def gate_viz(gates):
        return " ".join([f"{h_colors[i]}{gradient_char(g)}{Colors.RESET}" for i, g in enumerate(gates)])
    
    diagram = f"""
{Colors.BRIGHT_WHITE}╔═══════════════════════════════════════════════════════════════════════════════╗
║                        {Colors.BRIGHT_CYAN}R N A   E D I T I N G   L A Y E R{Colors.BRIGHT_WHITE}                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣{Colors.RESET}
║                                                                               ║
║   {Colors.BRIGHT_YELLOW}INPUT EMBEDDING{Colors.RESET}                                                          ║
║        │                                                                      ║
║        ▼                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────┐     ║
║   │  {Colors.CYAN}256-dimensional semantic vector → RNA Analysis{Colors.RESET}                    │     ║
║   └─────────────────────────────────────────────────────────────────────┘     ║
║        │                                                                      ║
║        ├──────────────────┬──────────────────┬──────────────────┐             ║
║        ▼                  ▼                  ▼                  ▼             ║
║   {Colors.BRIGHT_MAGENTA}┌────────┐{Colors.RESET}      {Colors.BRIGHT_CYAN}┌────────────┐{Colors.RESET}  {Colors.BRIGHT_YELLOW}┌───────────┐{Colors.RESET}  {Colors.BRIGHT_GREEN}┌──────────┐{Colors.RESET}    ║
║   {Colors.BRIGHT_MAGENTA}│CONFID. │{Colors.RESET}      {Colors.BRIGHT_CYAN}│ TEMPERATURE│{Colors.RESET}  {Colors.BRIGHT_YELLOW}│ PATHWAYS  │{Colors.RESET}  {Colors.BRIGHT_GREEN}│HEAD GATES│{Colors.RESET}    ║
║   {Colors.BRIGHT_MAGENTA}└────────┘{Colors.RESET}      {Colors.BRIGHT_CYAN}└────────────┘{Colors.RESET}  {Colors.BRIGHT_YELLOW}└───────────┘{Colors.RESET}  {Colors.BRIGHT_GREEN}└──────────┘{Colors.RESET}    ║
║                                                                               ║
║   {conf_color}Confidence: [{conf_bar}] {confidence*100:5.1f}%{Colors.RESET}                              ║
║   {temp_color}Temperature:[{temp_bar}] {temperature:5.2f}{Colors.RESET}                               ║
║                                                                               ║
║   {Colors.BRIGHT_YELLOW}═══ PATHWAY WEIGHTS ═══{Colors.RESET}                                               ║
║   {p_colors[0]}PERCEPTION  [{pathway_bar(pw[0])}] {pw[0]*100:5.1f}%{Colors.RESET} {"◄── PRIMARY" if primary_pathway == 0 else ""}    ║
║   {p_colors[1]}REASONING   [{pathway_bar(pw[1])}] {pw[1]*100:5.1f}%{Colors.RESET} {"◄── PRIMARY" if primary_pathway == 1 else ""}    ║
║   {p_colors[2]}ACTION      [{pathway_bar(pw[2])}] {pw[2]*100:5.1f}%{Colors.RESET} {"◄── PRIMARY" if primary_pathway == 2 else ""}    ║
║                                                                               ║
║   {Colors.BRIGHT_GREEN}═══ HEAD GATES (Personality Modulation) ═══{Colors.RESET}                          ║
║   ┌───────────┬───────────┬───────────┬───────────┐                           ║
║   │{h_colors[0]}Formality{Colors.RESET} │{h_colors[1]}Verbosity{Colors.RESET} │{h_colors[2]}Technical{Colors.RESET} │{h_colors[3]} Warmth  {Colors.RESET}│                           ║
║   │  {h_colors[0]}{hg[0]*100:5.1f}%{Colors.RESET}   │  {h_colors[1]}{hg[1]*100:5.1f}%{Colors.RESET}   │  {h_colors[2]}{hg[2]*100:5.1f}%{Colors.RESET}   │  {h_colors[3]}{hg[3]*100:5.1f}%{Colors.RESET}   │                           ║
║   ├───────────┼───────────┼───────────┼───────────┤                           ║
║   │{h_colors[4]} Hedging {Colors.RESET} │{h_colors[5]}Creativity{Colors.RESET}│{h_colors[6]}Directness{Colors.RESET}│{h_colors[7]} Empathy {Colors.RESET}│                           ║
║   │  {h_colors[4]}{hg[4]*100:5.1f}%{Colors.RESET}   │  {h_colors[5]}{hg[5]*100:5.1f}%{Colors.RESET}   │  {h_colors[6]}{hg[6]*100:5.1f}%{Colors.RESET}   │  {h_colors[7]}{hg[7]*100:5.1f}%{Colors.RESET}   │                           ║
║   └───────────┴───────────┴───────────┴───────────┘                           ║
║                                                                               ║
║        │                                                                      ║
║        ▼                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────┐     ║
║   │  {route_color}{route_icon} ROUTING DECISION: {route:12}{Colors.RESET}                                  │     ║
║   │  {Colors.DIM}{"Fast pattern matching" if route == "System 1" else "Deep deliberative reasoning":50}{Colors.RESET}         │     ║
║   └─────────────────────────────────────────────────────────────────────┘     ║
║                                                                               ║
{Colors.BRIGHT_WHITE}╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    return diagram

def generate_embedding(input_type):
    """Generate a synthetic embedding based on input type"""
    base = [random.gauss(0.5, 0.1) for _ in range(256)]
    
    if input_type == "code":
        # Technical inputs have strong early dimensions
        for i in range(50):
            base[i] = random.gauss(0.8, 0.1)
    elif input_type == "chat":
        # Social inputs have mid-range dimensions active
        for i in range(50, 150):
            base[i] = random.gauss(0.7, 0.1)
    elif input_type == "math":
        # Math has late dimensions
        for i in range(150, 256):
            base[i] = random.gauss(0.75, 0.1)
    elif input_type == "creative":
        # Creative is sparse but intense
        for i in range(0, 256, 3):
            base[i] = random.gauss(0.9, 0.05)
    
    return [max(0, min(1, v)) for v in base]

def analyze_input(embedding):
    """Analyze input using OCTO or simulation"""
    if bridge:
        result = bridge.analyze(embedding)
        return result
    else:
        # Simulation mode
        return {
            'confidence': random.uniform(0.4, 0.9),
            'temperature': random.uniform(0.5, 4.0),
            'pathway_weights': [random.uniform(0.2, 0.5) for _ in range(3)],
            'head_gates': [random.uniform(0.3, 0.8) for _ in range(8)]
        }

def normalize_pathways(weights):
    """Normalize pathway weights to sum to 1"""
    total = sum(weights)
    if total > 0:
        return [w/total for w in weights]
    return [0.33, 0.34, 0.33]

def run_demo():
    """Run the visualization demo"""
    clear_screen()
    print(draw_brain())
    
    # Load weights if available
    if bridge:
        try:
            bridge.load_weights('/home/worm/octotetrahedral-agi/rna_weights.pt')
            print(f"{Colors.GREEN}[✓] Loaded trained RNA weights{Colors.RESET}")
        except:
            print(f"{Colors.YELLOW}[!] Using untrained weights{Colors.RESET}")
    
    time.sleep(1)
    
    input_types = [
        ("code", "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"),
        ("chat", "Hey! How's it going? What did you do this weekend?"),
        ("math", "Calculate the integral of x^2 * e^x from 0 to infinity"),
        ("creative", "Write a haiku about quantum entanglement"),
        ("code", "SELECT users.* FROM users JOIN orders ON users.id = orders.user_id"),
        ("chat", "I'm feeling a bit down today, can we talk?"),
        ("math", "Prove that the square root of 2 is irrational"),
        ("creative", "Imagine a world where colors have sounds"),
    ]
    
    print(f"\n{Colors.BRIGHT_WHITE}{'═'*80}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  Starting real-time consciousness visualization...{Colors.RESET}")
    print(f"{Colors.BRIGHT_WHITE}{'═'*80}{Colors.RESET}\n")
    time.sleep(1.5)
    
    for input_type, sample_input in input_types:
        clear_screen()
        
        # Show input being processed
        print(f"\n{Colors.BRIGHT_WHITE}╔{'═'*78}╗{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}║{Colors.BRIGHT_YELLOW}  INPUT [{input_type.upper():8}]:{Colors.RESET} {sample_input[:55]:55} {Colors.BRIGHT_WHITE}║{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}╚{'═'*78}╝{Colors.RESET}\n")
        
        # Generate embedding and analyze
        embedding = generate_embedding(input_type)
        result = analyze_input(embedding)
        
        # Normalize pathway weights
        pw = normalize_pathways(result['pathway_weights'])
        primary = pw.index(max(pw))
        
        # Determine route
        route = "System 1" if result['confidence'] > 0.65 and result['temperature'] < 1.8 else "System 2"
        
        # Animate the analysis
        for frame in range(10):
            # Add some animation jitter
            animated_pw = [min(1, max(0, w + random.gauss(0, 0.02))) for w in pw]
            animated_hg = [min(1, max(0, g + random.gauss(0, 0.02))) for g in result['head_gates']]
            animated_conf = min(1, max(0, result['confidence'] + random.gauss(0, 0.01)))
            animated_temp = max(0.1, result['temperature'] + random.gauss(0, 0.05))
            
            move_cursor(1, 5)
            print(draw_pathway_diagram(
                animated_pw, 
                primary, 
                animated_hg,
                animated_conf,
                animated_temp,
                route
            ))
            
            time.sleep(0.1)
        
        # Hold final state
        time.sleep(2)
    
    # Final summary
    clear_screen()
    print(f"""
{Colors.BRIGHT_CYAN}
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║   {Colors.BRIGHT_GREEN}█▀▀ █▀▀█ █▀▄▀█ █▀▀█ █   █▀▀ ▀▀█▀▀ █▀▀{Colors.BRIGHT_CYAN}                              ║
    ║   {Colors.BRIGHT_GREEN}█   █  █ █ ▀ █ █▄▄█ █   █▀▀   █   █▀▀{Colors.BRIGHT_CYAN}                              ║
    ║   {Colors.BRIGHT_GREEN}▀▀▀ ▀▀▀▀ ▀   ▀ ▀    ▀▀▀ ▀▀▀   ▀   ▀▀▀{Colors.BRIGHT_CYAN}                              ║
    ║                                                                          ║
    ║   {Colors.BRIGHT_WHITE}OCTO RNA Editing Layer - Consciousness Router{Colors.BRIGHT_CYAN}                     ║
    ║                                                                          ║
    ║   {Colors.YELLOW}The bridge between stimulus and response,{Colors.BRIGHT_CYAN}                          ║
    ║   {Colors.YELLOW}where perception becomes understanding,{Colors.BRIGHT_CYAN}                            ║
    ║   {Colors.YELLOW}and understanding becomes action.{Colors.BRIGHT_CYAN}                                  ║
    ║                                                                          ║
    ║   {Colors.DIM}Press Ctrl+C to exit{Colors.BRIGHT_CYAN}                                              ║
    ╚══════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")

def interactive_mode():
    """Interactive mode where user can type inputs"""
    clear_screen()
    print(draw_brain())
    
    if bridge:
        try:
            bridge.load_weights('/home/worm/octotetrahedral-agi/rna_weights.pt')
            print(f"{Colors.GREEN}[✓] Loaded trained RNA weights{Colors.RESET}")
        except:
            print(f"{Colors.YELLOW}[!] Using untrained weights{Colors.RESET}")
    
    print(f"\n{Colors.BRIGHT_CYAN}Interactive Mode - Type anything to see how OCTO routes it!{Colors.RESET}")
    print(f"{Colors.DIM}(Type 'quit' to exit, 'demo' for automated demo){Colors.RESET}\n")
    
    while True:
        try:
            user_input = input(f"{Colors.BRIGHT_WHITE}> {Colors.RESET}").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'demo':
                run_demo()
                continue
            elif not user_input:
                continue
            
            # Simple heuristic for input type
            if any(kw in user_input.lower() for kw in ['def ', 'function', 'class ', 'import ', 'select ', 'var ', 'const ']):
                input_type = "code"
            elif any(kw in user_input.lower() for kw in ['prove', 'calculate', 'integral', 'equation', 'sum of']):
                input_type = "math"
            elif any(kw in user_input.lower() for kw in ['write', 'create', 'imagine', 'story', 'poem']):
                input_type = "creative"
            else:
                input_type = "chat"
            
            embedding = generate_embedding(input_type)
            result = analyze_input(embedding)
            
            pw = normalize_pathways(result['pathway_weights'])
            primary = pw.index(max(pw))
            route = "System 1" if result['confidence'] > 0.65 and result['temperature'] < 1.8 else "System 2"
            
            print(draw_pathway_diagram(
                pw, primary, result['head_gates'],
                result['confidence'], result['temperature'], route
            ))
            
        except KeyboardInterrupt:
            break
    
    print(f"\n{Colors.BRIGHT_CYAN}Goodbye!{Colors.RESET}\n")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            interactive_mode()
        else:
            run_demo()
    except KeyboardInterrupt:
        print(f"\n{Colors.BRIGHT_CYAN}Visualization ended.{Colors.RESET}\n")
