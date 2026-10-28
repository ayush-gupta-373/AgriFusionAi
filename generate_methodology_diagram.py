import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize figure with high resolution (300 DPI)
fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# Color Palette (Academic / Modern Tech)
BG_COLOR = "#F8FAFC"
LAYER_BG = "#FFFFFF"
HEADER_COLOR = "#0F172A"
BORDER_COLOR = "#CBD5E1"

BOX_COLORS = {
    "data": "#EFF6FF",       # Light blue
    "data_border": "#3B82F6",
    "ml": "#ECFDF5",         # Light emerald
    "ml_border": "#10B981",
    "gateway": "#F5F3FF",    # Light purple
    "gateway_border": "#8B5CF6",
    "client": "#FFFBEB",     # Light amber
    "client_border": "#F59E0B"
}

# Canvas background
fig.patch.set_facecolor(BG_COLOR)

# Title Banner
ax.text(8, 10.5, "AGRIFUSION AI: SYSTEM METHODOLOGY & ARCHITECTURE PIPELINE",
        ha='center', va='center', fontsize=18, fontweight='bold', color=HEADER_COLOR)
ax.text(8, 10.15, "End-to-End Multimodal Decision Support Framework for Precision Agriculture",
        ha='center', va='center', fontsize=11, color="#64748B", style='italic')

# Helper function to draw rounded boxes
def draw_card(ax, x, y, w, h, title, items, bg_color, border_color, title_color="#0F172A"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, ha='center', va='top',
            fontsize=10.5, fontweight='bold', color=title_color)
    
    # Bullet points
    item_y = y + h - 0.75
    for item in items:
        ax.text(x + 0.2, item_y, f"• {item}", ha='left', va='top',
                fontsize=8.5, color="#334155")
        item_y -= 0.32

# -------------------------------------------------------------
# LAYER 1: DATA INGESTION & EXTERNAL TELEMETRY (Bottom Layer: Y = 0.6 to 2.4)
# -------------------------------------------------------------
layer1_box = patches.FancyBboxPatch((0.5, 0.6), 15.0, 1.9, boxstyle="round,pad=0.1",
                                   facecolor=LAYER_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
ax.add_patch(layer1_box)
ax.text(0.8, 2.3, "LAYER 1: MULTIMODAL DATA INGESTION & REAL-TIME TELEMETRY",
        fontsize=10.5, fontweight='bold', color="#1E293B")

draw_card(ax, 0.8, 0.8, 3.3, 1.4, "Tabular Soil & Crop Data",
          ["Kaggle Crop Dataset (2,200 rows)", "Fertilizer Dataset (10,000 rows)", "Features: N, P, K, pH, Moisture"],
          BOX_COLORS["data"], BOX_COLORS["data_border"])

draw_card(ax, 4.4, 0.8, 3.3, 1.4, "Plant Leaf Pathology",
          ["PlantVillage Dataset (54,303 images)", "38 Classes across 14 Crop Species", "Healthy controls + Foliar Lesions"],
          BOX_COLORS["data"], BOX_COLORS["data_border"])

draw_card(ax, 8.0, 0.8, 3.5, 1.4, "Official Knowledge Corpora",
          ["PPQS Crop Packages (Rice, Wheat, Tomato)", "MNRE PM-KUSUM Guidelines (8.9 MB)", "PM-KISAN DBT Operational Handbooks"],
          BOX_COLORS["data"], BOX_COLORS["data_border"])

draw_card(ax, 11.8, 0.8, 3.4, 1.4, "External Live Web APIs",
          ["Open-Meteo API (Live Temp, Rain, Humid)", "Agmarknet API via data.gov.in (Mandis)", "OSM Nominatim Reverse Geocoding"],
          BOX_COLORS["data"], BOX_COLORS["data_border"])

# -------------------------------------------------------------
# LAYER 2: MACHINE LEARNING & AI REASONING ENGINES (FastAPI) (Y = 3.0 to 5.2)
# -------------------------------------------------------------
layer2_box = patches.FancyBboxPatch((0.5, 3.0), 15.0, 2.2, boxstyle="round,pad=0.1",
                                   facecolor=LAYER_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
ax.add_patch(layer2_box)
ax.text(0.8, 5.0, "LAYER 2: ASYNCHRONOUS ML & DEEP LEARNING ENGINES (Python / FastAPI Microservice)",
        fontsize=10.5, fontweight='bold', color="#065F46")

draw_card(ax, 0.8, 3.2, 3.3, 1.6, "Predictive Agronomy Models",
          ["Random Forest Classifier (99.2% F1)", "XGBoost Classifier for Fertilizer", "Automated Microclimate Feature Mapping", "5-Fold Stratified Cross-Validation"],
          BOX_COLORS["ml"], BOX_COLORS["ml_border"])

draw_card(ax, 4.4, 3.2, 3.3, 1.6, "EfficientNet-B0 Vision Engine",
          ["Compound Scaling: depth, width, resolution", "5.3M Parameters (79.3% < ResNet-50)", "Top-1 Accuracy: 98.7% | Latency: 32ms", "Out-of-Distribution Rejection (τ = 0.60)"],
          BOX_COLORS["ml"], BOX_COLORS["ml_border"])

draw_card(ax, 8.0, 3.2, 3.5, 1.6, "ChromaDB Vector Store (RAG)",
          ["Recursive Character Chunking (700/100)", "Local all-MiniLM-L6-v2 Embeddings", "Cosine Similarity Search over PPQS", "Strict Fact Grounding (<3.2% Hallucination)"],
          BOX_COLORS["ml"], BOX_COLORS["ml_border"])

draw_card(ax, 11.8, 3.2, 3.4, 1.6, "Cognitive Advisory Agent",
          ["Google Gemini 1.5 Flash LLM", "Prompt Engineering with Document Citations", "Hybrid Fallback Routing (Local vs General)", "Regional Multilingual Translation"],
          BOX_COLORS["ml"], BOX_COLORS["ml_border"])

# -------------------------------------------------------------
# LAYER 3: BACKEND GATEWAY & PERSISTENCE (Node.js + Express) (Y = 5.7 to 7.4)
# -------------------------------------------------------------
layer3_box = patches.FancyBboxPatch((0.5, 5.7), 15.0, 1.7, boxstyle="round,pad=0.1",
                                   facecolor=LAYER_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
ax.add_patch(layer3_box)
ax.text(0.8, 7.2, "LAYER 3: PLATFORM GATEWAY, AUTHENTICATION & BUSINESS LOGIC (Node.js + Express)",
        fontsize=10.5, fontweight='bold', color="#5B21B6")

draw_card(ax, 0.8, 5.9, 4.5, 1.2, "Security & Auth Gateway",
          ["Stateless JWT Token Authentication & bcrypt", "Secure Controller Proxies to Python FastAPI", "Hidden API Key Management (.env protection)"],
          BOX_COLORS["gateway"], BOX_COLORS["gateway_border"])

draw_card(ax, 5.6, 5.9, 4.5, 1.2, "Economics & Subsidy Engines",
          ["Live Agmarknet Mandi Pricing Controller", "Farm ROI Engine: Net Profit = (A·Y·P) - Costs", "Government Scheme Matcher (PM-KUSUM/KISAN)"],
          BOX_COLORS["gateway"], BOX_COLORS["gateway_border"])

draw_card(ax, 10.4, 5.9, 4.8, 1.2, "Persistence Layer (MongoDB Atlas)",
          ["Farmer Profiles, Field Polygons & GPS Coordinates", "Historical Soil Nutrient Logs (N-P-K Diary)", "Saved Disease Diagnoses & Chemical Prescriptions"],
          BOX_COLORS["gateway"], BOX_COLORS["gateway_border"])

# -------------------------------------------------------------
# LAYER 4: UNIVERSAL CLIENT INTERFACE (React.js SPA) (Y = 7.9 to 9.6)
# -------------------------------------------------------------
layer4_box = patches.FancyBboxPatch((0.5, 7.9), 15.0, 1.7, boxstyle="round,pad=0.1",
                                   facecolor=LAYER_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
ax.add_patch(layer4_box)
ax.text(0.8, 9.4, "LAYER 4: ACCESSIBLE CLIENT APPLICATION (React.js + Tailwind CSS Dashboard)",
        fontsize=10.5, fontweight='bold', color="#92400E")

draw_card(ax, 0.8, 8.1, 3.4, 1.2, "Spatial Farm Telemetry",
          ["Interactive Leaflet Map (All-India coverage)", "1-Click GPS Coordinate & Field Boundary Picker", "Auto-Populated 7-Day Microclimate Forecast"],
          BOX_COLORS["client"], BOX_COLORS["client_border"])

draw_card(ax, 4.5, 8.1, 3.4, 1.2, "Agronomic Advisory & Doctor",
          ["Soil Health & Fertilizer Calculator Form", "Drag-and-Drop Foliar Photo Diagnosis", "Instant CIBRC Chemical & Organic Remedies"],
          BOX_COLORS["client"], BOX_COLORS["client_border"])

draw_card(ax, 8.2, 8.1, 3.4, 1.2, "Market Economics & Trends",
          ["Live Mandi Price Comparison Table", "7-Day Commodity Price Fluctuation (Recharts)", "Estimated Net Profit Output in Rupees (₹)"],
          BOX_COLORS["client"], BOX_COLORS["client_border"])

draw_card(ax, 11.9, 8.1, 3.3, 1.2, "Inclusive Voice Interface",
          ["HTML5 Web Speech API (Speech-to-Speech)", "English & Regional Voice Queries", "Audio Playback of Grounded Gemini Advice"],
          BOX_COLORS["client"], BOX_COLORS["client_border"])

# -------------------------------------------------------------
# CONNECTING ARROWS & FLOW ANNOTATIONS
# -------------------------------------------------------------
arrow_props = dict(facecolor='#64748B', edgecolor='#475569', width=1.8, headwidth=6, headlength=7)

# Upward Flow: Data -> ML
ax.annotate('', xy=(2.45, 3.0), xytext=(2.45, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(6.05, 3.0), xytext=(6.05, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(9.75, 3.0), xytext=(9.75, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(13.5, 3.0), xytext=(13.5, 2.5), arrowprops=arrow_props)

# Upward/Bi-directional Flow: ML -> Gateway
ax.annotate('', xy=(3.0, 5.7), xytext=(3.0, 5.2), arrowprops=arrow_props)
ax.annotate('', xy=(7.85, 5.7), xytext=(7.85, 5.2), arrowprops=arrow_props)
ax.annotate('', xy=(12.8, 5.7), xytext=(12.8, 5.2), arrowprops=arrow_props)

# Bi-directional Flow: Gateway <-> Client
ax.annotate('', xy=(2.5, 7.9), xytext=(2.5, 7.4), arrowprops=arrow_props)
ax.annotate('', xy=(6.2, 7.9), xytext=(6.2, 7.4), arrowprops=arrow_props)
ax.annotate('', xy=(9.9, 7.9), xytext=(9.9, 7.4), arrowprops=arrow_props)
ax.annotate('', xy=(13.5, 7.9), xytext=(13.5, 7.4), arrowprops=arrow_props)

# Save high-resolution publication PNG
output_path = r"c:\AgriFusionAi\research_paper\methodology_architecture.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
print("High-resolution methodology diagram generated successfully at:", output_path)
