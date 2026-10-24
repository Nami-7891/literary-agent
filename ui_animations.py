# ui_animations.py

def update_tracker_ui(tracker, s1, s2, s3, s4):
    css_styles = """
    <style>
        /* Alignment containers */
        .tracker-wrapper { display: flex; justify-content: center; margin-top: 20px; }
        .tracker-container { display: flex; flex-direction: column; align-items: flex-start; }
        .step-container { display: flex; align-items: flex-start; margin-bottom: 0px; }
        
        /* Icons and Lines */
        .icon-box { width: 30px; display: flex; flex-direction: column; align-items: center; margin-right: 15px; }
        .spinner { width: 20px; height: 20px; border: 3px solid #444; border-top: 3px solid #fff; border-radius: 50%; animation: spin 1s linear infinite; }
        .check-icon { width: 22px; height: 22px; background-color: #fff; color: #000; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; }
        .error-icon { width: 22px; height: 22px; background-color: #e53935; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; }
        .pending-circle { width: 20px; height: 20px; border: 2px solid #444; border-radius: 50%; }
        
        /* The progressive filling line logic */
        .line-wrapper { position: relative; width: 2px; height: 35px; margin-top: 4px; margin-bottom: 4px; }
        .line-dashed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-left: 2px dashed #666; }
        .line-solid { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-left: 2px solid #fff; }
        .line-filling { position: absolute; top: 0; left: 0; width: 100%; background-color: #fff; animation: fill-down 2s infinite; }
        
        /* Typography (nowrap prevents the staircase glitch) */
        .text-box { font-family: sans-serif; font-size: 16px; padding-top: 2px; white-space: nowrap; color: #ddd; }
        .text-done { color: #fff; }
        .text-failed { color: #ff6b6b; }
        .text-pending { color: #666; }
        .agent-emoji { margin-left: 10px; font-size: 20px; display: inline-block; animation: bounce 1s infinite; vertical-align: middle; }
        
        /* Animations */
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fill-down { 0% { height: 0%; } 100% { height: 100%; } }
        @keyframes pulse-dot { 0% { opacity: 0.2; } 20% { opacity: 1; } 100% { opacity: 0.2; } }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
        
        .dot { animation: pulse-dot 1.4s infinite both; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
    """

    # Native emojis instead of external URLs (prevents broken image icons)
    emojis = {
        "planner": "📖",
        "analyzer": "🔍",
        "cartographer": "🗺️",
        "synthesizer": "✍️"
    }

    def render_step(name, present_text, past_text, state, emoji_icon, is_last=False):
        if state == "done":
            icon = "<div class='check-icon'>✓</div>"
            line = "" if is_last else "<div class='line-wrapper'><div class='line-solid'></div></div>"
            text = f"<span class='text-done'><b>{name}</b>&nbsp;{past_text}</span>"
            anim_element = ""
        elif state in ("failed", "error"):
            icon = "<div class='error-icon'>✕</div>"
            line = "" if is_last else "<div class='line-wrapper'><div class='line-solid'></div></div>"
            text = f"<span class='text-failed'><b>{name}</b>&nbsp;failed / unverified</span>"
            anim_element = ""
        elif state == "active":
            icon = "<div class='spinner'></div>"
            line = "" if is_last else "<div class='line-wrapper'><div class='line-dashed'></div><div class='line-filling'></div></div>"
            text = f"<span class='text-done'><b>{name}</b>&nbsp;{present_text} <span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
            anim_element = f"<span class='agent-emoji'>{emoji_icon}</span>"
        else: 
            icon = "<div class='pending-circle'></div>"
            line = "" if is_last else "<div class='line-wrapper'><div class='line-dashed'></div></div>"
            text = f"<span class='text-pending'><b>{name}</b>&nbsp;{present_text}</span>"
            anim_element = ""

        return f"<div class='step-container'><div class='icon-box'>{icon}{line}</div><div class='text-box'>{text}{anim_element}</div></div>"

    html = css_styles + "<div class='tracker-wrapper'><div class='tracker-container'>"
    html += render_step("Planner agent", "is reading the text", "has read the text", s1, emojis["planner"])
    html += render_step("Analyzer agent", "is extracting characters", "has extracted characters", s2, emojis["analyzer"])
    html += render_step("Cartographer agent", "is mapping locations", "has mapped locations", s3, emojis["cartographer"])
    html += render_step("Synthesizer agent", "is writing the dossier", "has written the dossier", s4, emojis["synthesizer"], is_last=True)
    html += "</div></div>"
    tracker.markdown(html, unsafe_allow_html=True)