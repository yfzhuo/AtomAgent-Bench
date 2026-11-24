#!/usr/bin/env python3
"""Preserve the original Fig1; redraw Fig2-4 using the supplied Markdown data."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for name in ['fig2_p0.py','fig3_pk.py','fig4_tasks.py']:
    subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,check=True)
print('Fig1 preserved; Fig2-4 regenerated.')
