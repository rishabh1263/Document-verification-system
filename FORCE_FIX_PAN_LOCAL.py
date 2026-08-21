from pathlib import Path
import py_compile
import shutil

p = Path(r"src\\documents\\pan\\pan_verification.py")

if not p.exists():
    raise SystemExit(f"FILE NOT FOUND: {p.resolve()}")

text = p.read_text(encoding="utf-8", errors="replace")

marker = 'print("========== VERIFY_PAN_CARD FINAL RESULT ==========")'
start = text.find(marker)

if start == -1:
    raise SystemExit("VERIFY_PAN_CARD FINAL RESULT marker not found.")

ret = text.find("return _fast_pan_validation(image)", start)

if ret == -1:
    raise SystemExit("return _fast_pan_validation(image) not found.")

line_start = text.rfind("\n", 0, start) + 1
line_end = text.find("\n", ret)
if line_end == -1:
    line_end = len(text)

new_block = """    result = _fast_pan_validation(image)

    print("========== VERIFY_PAN_CARD FINAL RESULT ==========")
    print("verified:", result.get("verified"))
    print("validation:", result.get("validation"))
    print("===================================================")

    return result"""

backup = Path(str(p) + ".before_debug_fix")
shutil.copy2(p, backup)

text = text[:line_start] + new_block + text[line_end:]
p.write_text(text, encoding="utf-8", newline="\n")

py_compile.compile(str(p), doraise=True)

check = p.read_text(encoding="utf-8", errors="replace")

print("==========================================")
print("PAN DEBUG BLOCK FIXED")
print("FILE:", p.resolve())
print("BACKUP:", backup.resolve())
print("OLD verified variable:", 'print("verified:", verified)' in check)
print("OLD validation variable:", 'print("validation:", validation)' in check)
print("CORRECT result assignment:", "result = _fast_pan_validation(image)" in check)
print("SYNTAX: PASS")
print("==========================================")