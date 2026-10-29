from pathlib import Path
import shutil

p = Path("app.py")
s = p.read_text()

start_marker = '                                if not positive_points:\n'
try_marker = '                                try:\n                                    with st.spinner(\n                                        f"Refining candidate {index + 1}..."\n'

start = s.find(start_marker)
if start == -1:
    raise SystemExit("ERROR: Could not find positive-point validation.")

try_start = s.find(try_marker, start)
if try_start == -1:
    raise SystemExit("ERROR: Could not find the refinement try block.")

except_marker = '                                except Exception as exc:\n'
except_start = s.find(except_marker, try_start)
if except_start == -1:
    raise SystemExit("ERROR: Could not find the refinement exception handler.")

# Find the end of the exception handler, immediately before result columns.
end_marker = '                            result_col1, result_col2 = st.columns(2)\n'
end = s.find(end_marker, except_start)
if end == -1:
    raise SystemExit("ERROR: Could not find the end of the refinement section.")

# The positive-points validation already has an if/else.
# Put the entire try/except block inside that else branch.
block = s[try_start:end]
lines = block.splitlines(keepends=True)

# The try block currently aligns with the "if not positive_points".
# Indent the complete try/except block one additional level.
indented_block = "".join(
    ("    " + line) if line.strip() else line
    for line in lines
)

backup = Path("app_before_refinement_guard.py")
if not backup.exists():
    shutil.copy2(p, backup)

s = s[:try_start] + indented_block + s[end:]
p.write_text(s)

print(f"Updated refinement guard. Backup: {backup}")
