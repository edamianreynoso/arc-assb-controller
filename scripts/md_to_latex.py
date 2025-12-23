#!/usr/bin/env python3
"""Convert main_draft.md to main.tex for arXiv submission."""

import re

def convert_markdown_to_latex(md_content):
    """Convert markdown content to LaTeX."""
    
    # LaTeX preamble
    preamble = r'''\documentclass[11pt]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{array}
\usepackage{caption}
\usepackage{longtable}
\usepackage{listings}
\usepackage{float}
\usepackage[margin=1in]{geometry}

\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  columns=fullflexible,
  keepspaces=true
}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents}

\author{
  J. Eduardo Dami\'an Reynoso \\
  Independent Researcher \\
  \texttt{edamianreynoso@gmail.com}
}

\date{14 December 2025}

\begin{document}

\maketitle

'''
    
    lines = md_content.split('\n')
    output_lines = []
    in_code_block = False
    code_lang = ""
    in_table = False
    table_lines = []
    skip_until_empty = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip metadata at the beginning
        if i < 10 and (line.startswith('**Author:**') or line.startswith('**Affiliation:**') or 
                       line.startswith('**Email:**') or line.startswith('**Date:**') or 
                       line.startswith('**Status:**') or line == '---'):
            i += 1
            continue
        
        # Skip title (already in preamble)
        if line.startswith('# Affective Regulation Core'):
            i += 1
            continue
            
        # Handle code blocks
        if line.startswith('```'):
            if in_code_block:
                output_lines.append(r'\end{lstlisting}')
                in_code_block = False
            else:
                in_code_block = True
                code_lang = line[3:].strip()
                if code_lang == 'bash':
                    output_lines.append(r'\begin{lstlisting}[language=bash]')
                elif code_lang == 'python':
                    output_lines.append(r'\begin{lstlisting}[language=python]')
                else:
                    output_lines.append(r'\begin{lstlisting}')
            i += 1
            continue
        
        if in_code_block:
            output_lines.append(line)
            i += 1
            continue
        
        # Handle HTML comments (labels)
        if '<!-- LABEL:' in line:
            match = re.search(r'<!-- LABEL:([^>]+) -->', line)
            if match:
                label = match.group(1)
                output_lines.append(f'\\label{{{label}}}')
            i += 1
            continue
        
        # Handle images
        if line.startswith('!['):
            match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if match:
                caption = match.group(1)
                path = match.group(2)
                output_lines.append(r'\begin{figure}[H]')
                output_lines.append(r'    \centering')
                output_lines.append(f'    \\includegraphics[width=0.9\\textwidth]{{{path}}}')
                # Check next line for caption
                if i + 1 < len(lines) and lines[i+1].startswith('*') and lines[i+1].endswith('*'):
                    cap_text = lines[i+1][1:-1]  # Remove asterisks
                    cap_text = escape_latex(cap_text)
                    output_lines.append(f'    \\caption{{{cap_text}}}')
                    i += 1
                output_lines.append(r'\end{figure}')
            i += 1
            continue
        
        # Handle figure captions (standalone)
        if line.startswith('*Figure') and line.endswith('*'):
            i += 1
            continue
        
        # Handle tables
        if '|' in line and not in_table:
            # Start of table
            in_table = True
            table_lines = [line]
            i += 1
            continue
        
        if in_table:
            if '|' in line:
                table_lines.append(line)
                i += 1
                continue
            else:
                # End of table
                output_lines.extend(convert_table(table_lines))
                in_table = False
                table_lines = []
                # Don't increment i, process current line
                continue
        
        # Handle headers
        if line.startswith('## Appendix '):
            match = re.match(r'## Appendix ([A-Z]): (.+)', line)
            if match:
                title = escape_latex(match.group(2))
                output_lines.append(f'\\section{{{title}}}')
            i += 1
            continue
        
        if line.startswith('### ') and re.match(r'### [A-Z]\.\d+', line):
            match = re.match(r'### [A-Z]\.\d+ (.+)', line)
            if match:
                title = escape_latex(match.group(1))
                output_lines.append(f'\\subsection{{{title}}}')
            i += 1
            continue
        
        if line.startswith('## ') and re.match(r'## \d+\.', line):
            match = re.match(r'## \d+\. (.+)', line)
            if match:
                title = escape_latex(match.group(1))
                output_lines.append(f'\\section{{{title}}}')
            i += 1
            continue
        
        if line.startswith('### ') and re.match(r'### \d+\.\d+', line):
            match = re.match(r'### \d+\.\d+ (.+)', line)
            if match:
                title = escape_latex(match.group(1))
                output_lines.append(f'\\subsection{{{title}}}')
            i += 1
            continue
        
        if line.startswith('#### ') and re.match(r'#### \d+\.\d+\.\d+', line):
            match = re.match(r'#### \d+\.\d+\.\d+ (.+)', line)
            if match:
                title = escape_latex(match.group(1))
                output_lines.append(f'\\subsubsection{{{title}}}')
            i += 1
            continue
        
        if line.startswith('### Figure S'):
            match = re.match(r'### (.+)', line)
            if match:
                title = escape_latex(match.group(1))
                output_lines.append(f'\\subsection{{{title}}}')
            i += 1
            continue
        
        if line.startswith('## Abstract'):
            output_lines.append(r'\begin{abstract}')
            i += 1
            # Collect abstract content
            while i < len(lines) and not lines[i].startswith('**Keywords:**') and not lines[i].startswith('---'):
                if lines[i].strip():
                    output_lines.append(convert_inline(lines[i]))
                i += 1
            output_lines.append(r'\end{abstract}')
            continue
        
        if line.startswith('**Keywords:**'):
            keywords = line.replace('**Keywords:**', '').strip()
            output_lines.append(f'\\textbf{{Keywords:}} {escape_latex(keywords)}')
            i += 1
            continue
        
        if line.startswith('## References'):
            output_lines.append(r'\bibliographystyle{unsrt}')
            output_lines.append(r'\bibliography{references}')
            i += 1
            # Skip reference list (handled by BibTeX)
            while i < len(lines) and not lines[i].startswith('## ') and not lines[i].startswith('---'):
                i += 1
            continue
        
        # Handle horizontal rules
        if line.strip() == '---':
            i += 1
            continue
        
        # Handle math blocks
        if line.strip() == '$$':
            output_lines.append(r'\begin{equation}')
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                eq_line = lines[i]
                if '\\label{' in eq_line:
                    output_lines.append(eq_line)
                else:
                    output_lines.append(eq_line)
                i += 1
            output_lines.append(r'\end{equation}')
            i += 1
            continue
        
        # Handle bullet points
        if line.startswith('- '):
            if not output_lines or not output_lines[-1].endswith(r'\begin{itemize}'):
                output_lines.append(r'\begin{itemize}')
            item_text = convert_inline(line[2:])
            output_lines.append(f'    \\item {item_text}')
            # Check if next line is not a bullet point
            if i + 1 < len(lines) and not lines[i+1].startswith('- '):
                output_lines.append(r'\end{itemize}')
            i += 1
            continue
        
        # Handle numbered lists
        match = re.match(r'^(\d+)\. (.+)$', line)
        if match and int(match.group(1)) <= 10:
            if not output_lines or not output_lines[-1].endswith(r'\begin{enumerate}'):
                output_lines.append(r'\begin{enumerate}')
            item_text = convert_inline(match.group(2))
            output_lines.append(f'    \\item {item_text}')
            # Check if next line is not a numbered item
            if i + 1 < len(lines):
                next_match = re.match(r'^(\d+)\. ', lines[i+1])
                if not next_match:
                    output_lines.append(r'\end{enumerate}')
            i += 1
            continue
        
        # Regular paragraph text
        if line.strip():
            output_lines.append(convert_inline(line))
        else:
            output_lines.append('')
        
        i += 1
    
    # Handle any remaining table
    if in_table and table_lines:
        output_lines.extend(convert_table(table_lines))
    
    # Add appendix marker before first appendix
    latex_content = '\n'.join(output_lines)
    latex_content = latex_content.replace(r'\section{Reproducibility}', r'\appendix' + '\n\n' + r'\section{Reproducibility}', 1)
    
    return preamble + latex_content + '\n\n\\end{document}\n'


def escape_latex(text):
    """Escape special LaTeX characters."""
    # Don't escape $ for math
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('#', r'\#')
    # Escape underscores not in math mode (simplified)
    # text = text.replace('_', r'\_')
    return text


def convert_inline(text):
    """Convert inline markdown to LaTeX."""
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    # Italic
    text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    # Escape special chars
    text = escape_latex(text)
    return text


def convert_table(table_lines):
    """Convert markdown table to LaTeX tabular."""
    result = []
    
    if len(table_lines) < 2:
        return result
    
    # Parse header
    header = table_lines[0]
    cols = [c.strip() for c in header.split('|') if c.strip()]
    num_cols = len(cols)
    
    # Create column spec
    col_spec = 'l' + 'c' * (num_cols - 1)
    
    result.append(r'\begin{table}[H]')
    result.append(r'\centering')
    result.append(r'\small')
    result.append(f'\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}')
    result.append(r'\toprule')
    
    # Header row
    header_cells = [escape_latex(c.strip()).replace('_', r'\_') for c in cols]
    result.append(' & '.join(header_cells) + r' \\')
    result.append(r'\midrule')
    
    # Data rows (skip separator line)
    for line in table_lines[2:]:
        if '---' in line:
            continue
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if cells:
            cell_text = [escape_latex(c).replace('_', r'\_') for c in cells]
            result.append(' & '.join(cell_text) + r' \\')
    
    result.append(r'\bottomrule')
    result.append(r'\end{tabular}')
    result.append(r'\end{table}')
    result.append('')
    
    return result


if __name__ == '__main__':
    with open('paper/main_draft.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    latex_content = convert_markdown_to_latex(md_content)
    
    with open('paper/main.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f'Conversion complete. Output: paper/main.tex')
    print(f'Lines in output: {len(latex_content.splitlines())}')
