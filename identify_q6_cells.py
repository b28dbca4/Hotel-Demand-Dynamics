#!/usr/bin/env python3
"""
Script để chạy lại tất cả cells Q6 trong notebook
"""

import json
import sys
from pathlib import Path

# Đọc notebook
notebook_path = Path('/home/lang-phu-quy/Documents/Programing for Data science/Hotel-Demand-Dynamics/notebooks/03_eda_businees.ipynb')

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Danh sách các cells Q6 cần chạy (bắt đầu từ A1)
q6_cells_to_run = []

# Tìm tất cả code cells từ Q6
in_q6 = False
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    # Bắt đầu Q6
    if 'Question 6 (Q6)' in source or 'A1) CHUẨN BỊ SEGMENT-LEVEL SUMMARY' in source:
        in_q6 = True
    
    # Nếu đang trong Q6 và là code cell
    if in_q6 and cell['cell_type'] == 'code':
        cell_id = cell.get('id', f'cell_{i}')
        snippet = source[:80].replace('\n', ' ')
        q6_cells_to_run.append({
            'index': i,
            'id': cell_id,
            'preview': snippet
        })
        print(f"Cell {i}: {cell_id[:10]}... - {snippet}...")

print(f"\nTotal Q6 code cells found: {len(q6_cells_to_run)}")
print("\nThese cells need to be executed in order in the Jupyter notebook.")
print("Please run them manually or use the notebook interface.")
