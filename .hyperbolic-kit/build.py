import os
import re

# Paths
KIT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(KIT_ROOT, "static")
IMPL_DIR = os.path.join(KIT_ROOT, "implementations")
UI_CSS = os.path.join(KIT_ROOT, "ui/theme.css")

# Shared modules in dependency order
SHARED_MODULES = [
    "core/transform.js",
    "core/poincare.js",
    "core/geodesic.js",
    "ui/overlay.js",
    "engine/canvas.js",
    "engine/interact.js",
]

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_js(code):
    """Simple regex stripper for ES modules to global script."""
    # Remove import lines (handling multi-line)
    # Match "import ... from ...;" across newlines
    code = re.sub(r'^\s*import\s[\s\S]*?from\s*[\'"].*?[\'"];', '', code, flags=re.MULTILINE)
    # Remove 'export' keyword but keep declarations
    code = re.sub(r'^\s*export\s+', '', code, flags=re.MULTILINE)
    return code

def build_implementation(name):
    print(f"Building {name}...")
    src_dir = os.path.join(IMPL_DIR, name)
    index_path = os.path.join(src_dir, "index.html")
    script_path = os.path.join(src_dir, "script.js")

    if not os.path.exists(index_path):
        print(f"  Skipping (no index.html)")
        return

    # 1. Bundle JS
    full_js = "// --- BUNDLED BY BUILD.PY ---\n"
    full_js += "(function(){\n"  # Wrap in IIFE to avoid global pollution (optional, but good)
    
    # Add Shared
    for mod in SHARED_MODULES:
        mod_path = os.path.join(KIT_ROOT, mod)
        if os.path.exists(mod_path):
            full_js += f"\n// --- {mod} ---\n"
            full_js += clean_js(read_file(mod_path))
    
    # Add Implementation
    if os.path.exists(script_path):
        full_js += f"\n// --- {name}/script.js ---\n"
        full_js += clean_js(read_file(script_path))

    full_js += "\n})();"

    # 2. Bundle CSS
    css_content = ""
    if os.path.exists(UI_CSS):
        css_content = read_file(UI_CSS)

    # 3. Process HTML
    html = read_file(index_path)
    
    # Inline CSS
    # Create style tag if strictly needed, or replace link
    # We look for <link rel="stylesheet" href="../../ui/theme.css">
    html = re.sub(
        r'<link rel="stylesheet" href=".*?theme\.css">', 
        f'<style>\n{css_content}\n</style>', 
        html
    )

    # Inline JS
    # Replace <script type="module" src="./script.js"></script>
    html = re.sub(
        r'<script type="module" src=".*?"></script>',
        f'<script>\n{full_js}\n</script>',
        html
    )

    # Save to static
    out_path = os.path.join(STATIC_DIR, f"{name}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> {out_path}")

def main():
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
        
    for item in os.listdir(IMPL_DIR):
        if os.path.isdir(os.path.join(IMPL_DIR, item)):
            build_implementation(item)

if __name__ == "__main__":
    main()
