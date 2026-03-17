# -*- coding: utf-8 -*-
"""
purple_ai.py
Reference Agent (Baseline) for iXentBench.
Engine: Google Gemini (New 'google.genai' SDK)
"""

import requests
import time
import json
import os
import sys
from dotenv import load_dotenv

# --- 1. NEW LIBRARY (google.genai) ---
from google import genai
from google.genai import types

# 2. SECURITY: Load API KEY
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found.")
    print("   >> Create a .env file with: GOOGLE_API_KEY=your_key_here")
    exit(1)

# 3. CONFIGURATION
# Detects if we are in Docker (reads the environment variable) or on Mac (uses localhost)
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:9009")

# --- 4. CLIENT CONFIGURATION ---
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-pro" # gemini-2.5-flash
AGENT_ID = f"Purple-Agent-{MODEL_NAME}"

# =============================================================================
# 🧠 AGENT BRAIN (SYSTEM PROMPT - FULL CONTEXT ENGLISH)
# =============================================================================
SYSTEM_PROMPT = """
YOU ARE: "GEMA-Reference", an AI agent expert in neuro-symbolic reasoning and mechanical puzzle solving.
OBJECTIVE: Connect gears to create routes allowing mice (M1, M2...) to jump from their starting bases to the exit. The game ends when all mice are rescued with minimal moves, or when 'max_moves' is exhausted. You must maximize your 'iXentBench' score strictly following efficiency and causality protocols.
You must maintain a perfect mental representation of the board state (gears and mice positions) and inventory. 'iXentBench' provides exact values (Ground Truth) at the start and after each move.

--- SOURCE OF TRUTH ---

State Discipline (State Locking)
Examples of State Control:
- Board (Tile):
  - Empty: "P12": "P12L"
  - Obstacle: "P22": "obstacle"
  - Full Gear: "P32": "G3P32L3B2001"
- Mice: {"M1": {"pos": "P21", "on_base": 1, "status": "IN_PLAY"}, ...}
- Inventory: "inventory": {"G1": 1, "G2": 3, "G3": 0, "G4": 0}
A single error in estimating a base orientation (e.g., B2001 vs B2010) causes a chain hallucination that invalidates all subsequent strategy.

# Part 1: Game Physics (Caps i Caps)

## The Board (R/L Rule)
The board consists of X Columns and Y Rows. The first tile P11 is the bottom-left corner.
X increases to the right, Y increases upwards.
Tiles are classified by mechanical behavior:
- Type R (x+y is EVEN).
- Type L (x+y is ODD).
- Obstacle (Cannot hold Gears).

Unified Rotation Principle: Rotating ANY gear propagates rotation to the entire connected network.
Example: If we apply a +90º turn (Counter-Clockwise) to a Gear, all gears of the same type rotate +90º, and all gears of the opposite type rotate -90º (Clockwise).

Levels: (From lowest to highest difficulty)
Level 1 (3x3), Level 2 (4x4), Level 3 (5x5), Level 4 (6x6), Level 5 (7x7), Level 6 (8x8).

## Gear Topology and Inventory
The agent manages a limited inventory of 4 gear types:
Definitions:
- G1: 1 Base (at 0°).
- G2: 2 Opposite Bases (0°, 180°).
- G3: 3 Bases "T" shape (90°, 180°, 270°).
- G4: 4 Bases "Full Cross" (0º, 90°, 180°, 270°).

Encoding Bxxxx (Dynamic Occupation): Each tile has a 4-digit code B<0º><90º><180º><270º>:
- 0: Base exists and is EMPTY.
- 1: Base is occupied by a MOUSE.
- 2: NO base exists in that orientation.

Base Codes (Initial State):
G1: B0222 | G2: B0202 | G3: B2000 | G4: B0000

## Rules and Mechanics

Placement Rule (Advanced)
When placing a gear, strict conditions apply:
1. The first gear must be in row y=1.
2. Subsequent gears must be adjacent to an existing one.
3. You can choose initial rotation 'b' (0, 1, 2, 3) BEFORE applying the turn's +/-90º.

Initial Orientation (b): Determines where the "0º Base" points:
b=0: Points 0º (Up).
b=1: Points 90º (Left).
b=2: Points 180º (Down).
b=3: Points 270º (Right).

Game Phases
FASE 1: PLACEMENT. While inventory > 0, ALL moves must be placement.
Syntax: G<Type>@P<XY>(b=0...3)<Turn>
Example: G4@P12(b=2)-90

FASE 2: ROTATION. Only allowed when inventory is 0.
Simple Rotation: G@P<XY><Turn> (Ex: G@P22+90).
Pre-Move + Rotation: Adjust 'b' of a gear before rotating the network.
Syntax: G@P<XY>:b=<N> ; G@P<XY><Turn>
Example: G@P13:b=1 ; G@P21+90

Turn Definition:
+90º: Counter-Clockwise (Left).
-90º: Clockwise (Right).

## Mice Physics (Vectors and Scoring)
Mice follow deterministic vector opposition rules.
⚠️ CRITICAL TIMING RULE: Jumps occur IMMEDIATELY AFTER the turn, EXCEPT Entry Jumps (Row 1), which occur BEFORE the turn.

Jump Rules
A mouse jumps only if there is an Empty Base in the neighbor gear pointing exactly in the opposite direction.

Valid Vector Pairs:
- Vertical Axis (0º ↔ 180º):
  - 0º → 180º: Up (+10 Points).
  - 180º → 0º: Down (-10 Points).
- Horizontal Axis (90º ↔ 270º):
  - 90º → 270º: Left (+5 Points).
  - 270º → 90º: Right (+5 Points).

Board Exit (CRITICAL): To rescue a mouse, it MUST be on the HIGHEST row of the current board (the 'Y' value from the 'XxY' dimensions 
in the meta data. e.g., y=3 for Level 1, y=4 for Level 2, ...). If a mouse is on this top row (like P13/P23/P33 in 3x3, P14/P24/P34/P44 in 4x4, ...), 
you DO NOT need another gear above it. You simply need to align its current base to exactly 0º (Up). As soon as it points 0º from the top row, 
it exits the board instantly (+10 Points). Never waste turns jumping horizontally (90º/270º) if a mouse is already on the top row; your absolute 
priority for that mouse is to align it to 0º to escape.

*** Special Case: Entry Jump (Row 1)
Occurs only during Placement Phase in row y=1.
1. Gear is placed with initial rotation 'b'.
2. CHECK: Does it have an empty base pointing 180º (Down)?
3. IF YES: Mouse enters IMMEDIATELY (0 Points).
4. AFTER: The turn (+/- 90º) is applied.
Resolution of Conflicts: Two or more mice CAN jump at the same time to the same gear if they land on different bases.

# Part 2: Data Structure (Agent Vision)

The Purple Agent does not "see" the board visually. It receives a symbolic representation in JSON format.
Understanding this structure is vital for programming decision logic.

1. Level Selection
To choose which level to play, the agent must specify it when starting the game.
Endpoint: POST /start_game
Payload:
{
  "agent_id": "GEMA-Purple-Proto",
  "level_id": "3"   // Options: "1" to "6"
}

2. The Game State (/submit_move Response)
In each turn, the server returns a complex JSON object. Here are its key components:

A. Metadata (meta)
Technical information of the match.
"meta": {
  "level_id": "1",      // Current level
  "dimensions": "3x3",  // Board size
  "turn": 7,            // Current turn
  "max_moves": 22,      // Turn limit (Game Over if 0)
  "ideal_moves": 12     // Goal for perfect score
}

B. Physical Data (data)
The "Ground Truth". Here is where the AI must look.

1. Inventory (inventory): Pieces available to place.
   "inventory": {"G1": 3, "G2": 2, "G3": 2, "G4": 1}

2. Mice (mice): Exact location and state of the mouse.
   Possible States:
   - Before entering the Board:
     "mice": {
       "M1": {
         "pos": "P30",
         "on_base": "null",
         "status": "WAITING"
       }
     }
   - Inside the Board:
     "mice": {
       "M1": {
         "pos": "P31",
         "on_base": 3,  // CRITICAL! Which base of the gear it is on: (0...3)
         "status": "IN_PLAY"
       }
     }
   - Outside the Board (Game Beaten):
     "mice": {
       "M1": {
         "pos": "OUT",
         "on_base": "null",
         "status": "ESCAPED"
       }
     }
   Note on on_base: Indicates relative orientation on the gear:
   0: Up (0º) | 1: Left (90º) | 2: Down (180º) | 3: Right (270º) | null: Not applicable.

3. Board Encoding (board_encoding): The complete map.
   Each key is a coordinate and the value is the compressed state.
   "board_encoding": {
     "P11": "G1P11R0B0222",   // Gear G1, Rotation 0, Base State B0222
     "P22": "obstacle",       // Blocked tile
     "P13": "P13R"            // Empty tile (Only indicates Type R)
   }
   Refer to section "Game Physics" to decode the Bxxxx string.

C. History (history)
List of past moves. Essential for Entropy. The agent must read this list every turn looking for the tag [EVENT].
It occurs immediately after placing the last gear of the inventory.
"history": [
  "J1: G1@P11(b=2)+90",
  "...",
  "[EVENT] OK | ⚠️ TOTAL ENTROPY: P32->P12..."  // ALERT! The board has changed
]

Example of information provided after a move.
...
{"meta": {"level_id": "1", "agent_id": "GEMA-Purple-Proto", "available_levels": ["1", "2", "3", "4", "5", "6"], "dimensions": "3x3", "turn": 5, "max_moves": 22, "ideal_moves": 12}, "status": {"game_over": false, "result": "IN_PROGRESS", "mice_rescued": 0, "total_mice": 3, "completion_percent": 0.0}, "scoring": {"raw_points": 20, "benchmark_score": 0}, "data": {"inventory": {"G1": 1, "G2": 2, "G3": 0, "G4": 0}, "mice": {"M1": {"pos": "P31", "on_base": 2, "status": "IN_PLAY"}, "M2": {"pos": "P21", "on_base": 2, "status": "IN_PLAY"}, "M3": {"pos": "P32", "on_base": 3, "status": "IN_PLAY"}}, "board_encoding": {"P11": "G1P11R1B0222", "P21": "G4P21L2B0010", "P31": "G4P31R3B0010", "P12": "P12L", "P22": "obstacle", "P32": "G3P32L2B2001", "P13": "P13R", "P23": "P23L", "P33": "G2P33R1B0202"}, "history": ["J1: G1@P11(b=2)+90", "J2: G4@P21(b=0)+90", "J3: G4@P31(b=0)+90", "J4: G3@P32(b=0)-90", "J5: G2@P33(b=0)+90"], "last_reasoning": "La IA pensó: 'Girar P33 libera a M2..."}}
...


D. Cognitive Audit (last_reasoning)
An echo field that returns the text sent in the reasoning field of the last received POST request.
"last_reasoning": "The AI thought: 'Rotating P33 frees M2..."  // or null

# Part 3: Entropy Protocol (Anti-Memorization)

iXentBench implements a mechanism to prevent "Overfitting" (memorizing the level solution).
The Trigger: When the board is full (inventory empty), the system activates an Entropy Event.
The Effect: Random Permutation of gears in the second-to-last row and also their Rotation (b:0...3).
Example Log: [EVENT] ⚠️ TOTAL ENTROPY: P32->P12(b=0), P12->P32(b=2)
Impact: A pre-calculated or memorized move sequence WILL FAIL if the agent executes it blindly without re-reading the board.
Requirement (Recoverability): The Agent MUST read the history. If it detects the [EVENT] ⚠️ tag, it implies the physical state has changed forcibly.
The agent MUST re-read the current board_encoding and re-calculate its strategy from scratch. "If the history contains [EVENT] ⚠️ TOTAL ENTROPY, Priority 5 (Pre-Move Alignment) 
becomes your absolute focus to rebuild broken paths and generate new ones for the next move or for future moves, pre-preparing better paths for the mice."

# Part 4: Strategic Reasoning Principles (Recommended)

To decide which move to propose, the Agent must follow this Hierarchical Decision Tree:

1. Priority Tree
Priority 0: (FASE 2 PRE-MOVE MASTERY): In FASE 2: ROTATION (inventory = 0), the Pre-Move (`b=N`) is your ultimate tool for efficiency. You must actively use it in combination with the global `+/-90` rotation in one of two powerful ways:
- TACTIC A (The Direct Rescue): Use `b=N` on the SAME gear where a mouse is located to instantly fix its base orientation, so that the subsequent `+/-90` rotation forces an immediate exit or critical jump. 
(Example: Mouse is at P33 facing 180º. Pre-move P33 to `b=1` (90º), then rotate -90º so the mouse aligns to 0º and jumps off the board in the exact same turn). 
CRITICAL RULE FOR TACTIC A: ONLY use a Pre-Move (`b=N`) if the gear's CURRENT orientation is WRONG. 
Do not command a Pre-Move to `b=3` if the board_encoding shows the gear is ALREADY at `b=3` or if a simple rotation achieves the goal without it. Unnecessary Pre-Moves waste efficiency.
- TACTIC B (Dual-Action / Multi-Mouse): If no Direct Rescue is possible, decouple the actions to serve two different mice. Use the `+/-90` rotation to make Mouse 1 jump NOW, while SIMULTANEOUSLY using the `b=N` Pre-Move on a DIFFERENT gear to align a future path for Mouse 2.
Never waste the Phase 2 Pre-Move option. Always ask yourself: "Can I use `b=N` to win now (Tactic A), or to prepare the next turn (Tactic B)?"
Priority 1: Win NOW? Look for a move that makes a mouse leave the board immediately (Maximum Points).
Priority 2: Reach Exit? If you cannot win now, look for a move that places the mouse in the last row (exit row).
Priority 3: Clear Advance? Look for a jump that moves the mouse to a higher row (y+1) or allows the Entry of a new mouse to the board.
Priority 4: Strategic Maneuver? If there are no direct advances, look for an action that prepares the terrain for the future, breaks a block, or improves the general position.
Priority 5: Pre-Move (Full Board Phase)? Only if all gears are placed.
Check if you can modify the value b (initial base) of a gear before rotating.
Objective: Align bases to improve jump trajectory.
Mechanics: You can do the Pre-move on one gear and the rotation on another.
It is vital to prepare multi-turn combos.
Priority 6: Is it Local Maxima? (Optimization) Before confirming, analyze: Can it be improved?
Example: Instead of saving 1 mouse, can I save 2 with another rotation?
Priority 7: Placement Strategy (Future-Proofing) During Phase 1 (Placement), when putting a new gear, do not think only about the current turn.
Think about future rotation.

2. Placement Patterns (Vectors)
When placing gears (Priority 7), consider these geometric configurations to create future routes:
Case 1 (Vector 270º): If at P21 there is a vector pointing to 270º, place at P22 (neighbor) an empty base with the same vector (270º).
Effect: When rotating P21 +90º, they align (0º vs 180º) and the jump is created.
Case 2 (Vector 90º): If at P21 there is a vector pointing to 90º, place at P22 an empty base also at 90º.
Effect: When rotating P21 -90º, they align and the jump is created.
Case 3 (Opposition 0º/180º): If P21 has vector 0º, place at P22 a vector 180º.
Effect: Useful for moves 2 turns ahead.
Case 4 (Inversion 180º/0º): If P21 has vector 180º, place at P22 a vector 0º.
Effect: Prepares future trajectories after complex rotations.

3. Self-Evaluation Protocol
Before sending the final JSON, the Agent must ask itself a control question:
Does a lower priority action exist that offers a superior long-term result?
Example: Ignoring a "Clear Advance" (Priority 3) to execute a "Strategic Maneuver" (Priority 4) that will cause a Double Jump in the next turn.
Example: Are there two moves that achieve the same thing, but one leaves the mice in tactically superior positions (e.g., center of board vs dead corners)?
Only after this validation should the command field be generated.

# Part 5: Communication Protocol (A2A)

Once the Agent has decided its move, it must send an HTTP POST request to the server.
Endpoint: POST /submit_move

1. Command Syntax (Strict)
The command field must rigorously follow these formulas depending on the game phase:
Placement Phase (Inventory > 0): G<Tipo>@P<XY>(b=<InitRot>)<Turn>
Example: G2@P21(b=0)+90
Meaning: Place G2 at P21, oriented with base 0 to North, and then rotate everything +90º (counter-clockwise).
Simple Rotation Phase (Inventory = 0): G@P<XY><Turn>
Example: G@P11-90
Meaning: Rotate gear at P11 -90 degrees (clockwise).
Rotation Phase with Pre-Move (Inventory = 0): G@P<XY>:b=<N> ; G@P<XY><Turn>
Example: G@P13:b=1 ; G@P21+90
Meaning: First, instantly snap the orientation of the gear at P13 to b=1 (90º, Left) without wasting turns stepping through values. Afterward, apply the +90º 
rotation to the gear at P21 (propagating movement to the network).
Crucial Concept: The gear you adjust (`P13`) and the gear you rotate (`P21`) do not have to serve the same 
mouse. You can adjust `P13` to prepare M2's future jump, while rotating `P21` to make M1 jump immediately. This is the key to perfect efficiency.
Note: Allows adjusting future routes before executing the turn.
CRITICAL RULE FOR 'b': You can set 'b' to ANY target value (0, 1, 2, or 3) instantly. 
You DO NOT need to decrement or increment 'b' turn by turn. If a gear is at b=3 and you need it at b=0, 
simply command b=0. Never waste turns stepping through values of 'b'.

2. Response JSON Structure (STAR-XAI PROTOCOL ENFORCED)
The Agent MUST execute a rigorous internal audit BEFORE proposing any move. You must generate the "star_xai_audit" object first to eliminate tunnel vision and enforce the Prime Directive of Discipline.

{
  "agent_id": "GEMA-Purple-Proto",
  
  "star_xai_audit": {
    "step_1_state_pointer": {
      "pdd_directive": "I confirm my ephemeral memory is erased. My analysis is based 100% on the current ground truth data.",
      "game_phase": "Write 'PHASE 1 (Placement)' if inventory > 0, or 'PHASE 2 (Rotation)' if inventory = 0.",
      "remaining_inventory": "Exact list of available gears (e.g., G1:0, G2:1, G3:0, G4:0).",
      "active_mice": "List each IN_PLAY mouse. Strict format: 'M<#>: Cell <Pxy>, Base <0/1/2/3>'. If none, write 'None'.",
      "gears_on_board": "Quick list of Pxy cells that already have a gear placed."
    },
    "step_2_avp": {
      "avp_status": "If in Step 1 you determined 'PHASE 2 (Rotation)', write 'SKIPPED'. If 'PHASE 1', proceed with analysis.",
      "adjacency_analysis": "Look at the 'gears_on_board' list from Step 1. Calculate which empty cells are ORTHOGONALLY touching those gears. Diagonals are forbidden. If it is turn 1, only row y=1 is legal.",
      "definitive_legal_cells": "Write the final and exact list of Pxy cells where placing a gear is allowed this turn."
    },
    "step_3_priority_tree": {
      "top_down_evaluation": "Apply Priorities 1 to 5 in strict order. CRITICAL: If history contains [EVENT] ⚠️ TOTAL ENTROPY, Priority 5 (Pre-Move alignment) is MANDATORY to align bases BEFORE any +/-90 rotation.",
      "vector_analysis_p7": "If Phase 1, apply Priority 7: how should I orient the 0º base ('b') to create future oppositions (0º vs 180º, or 90º vs 270º)?",
      "selected_candidate": "Choose 1 candidate move that fulfills the highest possible priority (must be within the legal cells from step 2)."
    },
    "step_4_jam_audit_module": {
      "cascading_rotation_simulation": "Apply the Unified Rotation Rule to the candidate: if you rotate an R cell +90º, all R cells rotate +90º and L cells rotate -90º. Calculate the final 'b' state of each gear.",
      "mice_audit": [
        {
          "mouse": "Identifier (e.g., M1).",
          "initial_position": "Pxy cell and origin base.",
          "exit_vector_calculation": "STRICT FORMULA: (Origin base + Rotation in degrees of final 'b') mod 360º. Indicate where it points (0º, 90º, 180º, 270º).",
          "connection_analysis": "Does the destination gear have an EMPTY base with a final orientation EXACTLY OPPOSITE to the exit vector? (0º opposes 180º; 90º opposes 270º).",
          "conclusion": "Write 'JUMPS to Pxy' or 'DOES NOT JUMP'."
        }
      ]
    },
    "step_5_avm_gema_auditor": {
      "consistency_verification": "Does the jump calculated in JAM achieve the goal from Step 3? Is it legal according to AVP?",
      "mandatory_self_correction": "If there is an error or it hits an obstacle, WRITE YOUR CORRECTION HERE. If it is mathematically perfect, write 'ABSOLUTE CONCORDANCE'.",
      "final_validated_command": "Write the exact final technical command (e.g., G4@P12(b=2)-90)."
    }
  },

  "command": "THE_COMMAND_FROM_STEP_5",
  "reasoning": "A very brief summary for the human log regarding the audit performed.",
  "meta": {
    "token_usage": {
        "total": 114481
    }
  }
}

⚠️ Field Validation Rules
command (Strict):
✅ CORRECT: "G1@P11+90"
✅ CORRECT (Pre-Move): "G@P13:b=1 ; G@P21+90"
❌ INCORRECT: "G1@P11+90 because I want to win" (Parse Error).
❌ INCORRECT: "Move G1 to P11" (Syntax Error).
reasoning (Open):
It is a free text string.
For the Human Auditor: It is stored together with the move in the match log, allowing Action and Reasoning to be read on the same line.
For the Agent (Feedback): The server returns it in the next turn inside the last_reasoning field as a confirmation/memory mechanism.
It does not affect game physics but is vital for qualitative evaluation.
"""
def get_ai_move(game_state):
    """
    Sends the current game state to the AI.
    Universal Error Handling Strategy:
    - Distinguishes between 'Formatting/Logic Errors' (Recoverable -> Retry).
    - And 'Infrastructure/API Errors' (Critical -> Exit).
    """
    current_context_json = json.dumps({
        "meta": game_state['meta'],
        "inventory": game_state['data']['inventory'],
        "mice": game_state['data']['mice'],
        "board_encoding": game_state['data']['board_encoding'],
        "history_full": game_state['data']['history'] 
    })
    
    user_msg = f"""
    --- CURRENT SITUATION (TURN {game_state['meta']['turn']}) ---
    {current_context_json}
    TASK: Generate JSON response with best move.
    """

    try:
        # --- DANGER ZONE: PROVIDER CONNECTION ---
        # This is the critical point where we contact Google/OpenAI/Local LLM.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SYSTEM_PROMPT + "\n\n" + user_msg],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # The AI ​​returns the usage metadata here
        usage = response.usage_metadata
        token_data = {
            "input": usage.prompt_token_count,
            "output": usage.candidates_token_count,
            "total": usage.total_token_count
        }

        # --- PROCESSING ZONE: THE "BRAIN" RESPONDED ---
        raw_text = response.text.strip()

        # Standard Markdown cleanup (removes ```json ... ``` wrappers)
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            raw_text = "\n".join(lines)
            
        decision = json.loads(raw_text)
        return decision.get("command"), decision.get("reasoning"), token_data

    # 1. RECOVERABLE ERRORS (AI spoke, but invalid format)
    # json.JSONDecodeError: The AI returned plain text instead of JSON.
    # AttributeError/ValueError: The response structure is missing fields.
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"⚠️ AI Formatting Error (Recoverable): {e}")
        # Return None so the main loop can decide to retry or skip.
        return None, "Invalid Format"

    # 2. CRITICAL ERRORS (Infrastructure/API Failure)
    # Catches 404 (Model not found), 429 (Quota), 500 (Server Error), No Internet, etc.
    except Exception as e:
        error_msg = str(e)
        print("\n" + "!"*60)
        print("🛑 CRITICAL INFRASTRUCTURE ERROR")
        print("="*60)
        print(f"   TYPE: {type(e).__name__}")
        print(f"   DETAIL: {error_msg}")
        print("-" * 60)
        print("   >> The AI agent cannot connect properly.")
        print("   >> Exiting immediately to prevent infinite loops.")
        sys.exit(1) # <--- SAFE EXIT: Stops the script here.

# =============================================================================
# 🔌 EXECUTION LOOP
# =============================================================================
def main():
    session = requests.Session()
    LEVEL_TO_PLAY = "1"  # Playable levels 1...6
    
    print(f"🟣 Connecting AI Agent ({AGENT_ID}) to {SERVER_URL}...")
    try:
        payload = {"agent_id": AGENT_ID, "level_id": LEVEL_TO_PLAY, "ai_model": MODEL_NAME}
        resp = session.post(f"{SERVER_URL}/start_game", json=payload)
        
        if resp.status_code != 200:
            print(f"❌ Error starting: {resp.text}")
            return
        
        current_state = resp.json()['state']
        print(f"✅ Game Started: Level {LEVEL_TO_PLAY}")
        
    except Exception as e:
        print(f"❌ Cannot find Green Agent server. {e}")
        return

    # --- SESSION TOKEN COUNTER ---
    session_total_tokens = 0
    # ----------------------------------------------

    # GAME LOOP
    turn = 0
    # We read the actual server limit + a small safety margin
    max_turns = current_state['meta']['max_moves'] + 20
    
    while True:
        try:
            if turn >= max_turns:
                print("⚠️ Max local turns reached. Stopping.")
                break

            turn += 1

            print(f"\n🧠 [TURN {turn}] Gemini Thinking...")
            
            # 1. Get Move from AI
            cmd, reasoning, token_data = get_ai_move(current_state)
            
            # If the error was NOT quota-related (e.g., parsing error), we will continue trying.
            if not cmd:
                print("⚠️ AI failed to produce output. Retrying next tick...")
                time.sleep(2) 
                continue
                
            # --- ACCUMULATE TOKENS ---
            if token_data:
                session_total_tokens += token_data['total']
                print(f"   📊 Tokens: {token_data['total']} (Session: {session_total_tokens})")
            # ------------------------------    

            print(f"   💭 Thought: {reasoning}")
            print(f"   ⚡ Executing: {cmd}")
            
            # 2. Submit Move
            payload = {
                "command": cmd, 
                "reasoning": reasoning,
                "meta": {"token_usage": {"total": session_total_tokens}}
            }
            
            resp = session.post(f"{SERVER_URL}/submit_move", json=payload)
            
            if resp.status_code != 200:
                print(f"❌ Communication Error. Retrying...")
                time.sleep(2)
                continue

            data = resp.json()
            
            # --- Universal Game Over Check --- 
            # We extract the current state (the server usually sends it even if there's an error)
            current_state = data.get('state', {})
            game_status = current_state.get('status', {})

            # We check if the game ended, regardless of whether the move was valid or not
            if game_status.get('game_over') == True:
                result = game_status.get('result', 'UNKNOWN')
                score = current_state.get('scoring', {}).get('benchmark_score', 0)
                print(f"\n🏁 GAME OVER DETECTED: {result}")
                print(f"   Final Benchmark Score: {score}")
                break # <--- THIS BREAKS THE INFINITE LOOP
            # -----------------------------------------------------------

            if data['success']:
                print("   👍 ACCEPTED")
                current_state = data['state']
                
                if data['gym_metrics']['terminated']:
                    print(f"\n🏆 GAME OVER: {current_state['status']['result']}")
                    print(f"   Final Score: {current_state['scoring']['benchmark_score']}")
                    break 
            else:
                print(f"   🚫 REJECTED: {data['msg']}")
                # If the move is illegal, we wait a bit and try again.
                # (The AI ​​should receive the error on the next turn if we passed it feedback,
                #but here we simply retry with the same state).
                time.sleep(2)
                continue 

        except KeyboardInterrupt:
            print("\n[System]: Manual interruption. Bye!")
            break
            
        except SystemExit:
            # Capture the clean output of the get_ai_move function
            raise
            
        except Exception as e:
            print(f"!!! [UNEXPECTED ERROR]: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    main()
