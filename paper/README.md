# Bubble Transformer V5: Focus-Inspired Sinkhorn Token Grouping for Causal Language Model Attention

## arXiv Submission Package

This directory contains the LaTeX source for the arXiv preprint.

### Files

- `main.tex` - Main paper
- `supplementary.tex` - Supplementary material
- `build_arxiv.py` - Script to build submission package
- `references.bib` - Bibliography (if using BibTeX)

### Building

```bash
# Compile LaTeX
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Build arXiv package
python build_arxiv.py
```

### Submission

1. Go to https://arxiv.org/submit
2. Upload `submission.tar.gz`
3. Select category: `cs.CL` (Computation and Language)
4. Add comments: "11 pages, 3 figures, 5 tables"

### Citation

```bibtex
@article{bubble2026,
  title={Bubble Transformer V5: Focus-Inspired Sinkhorn Token Grouping for Causal Language Model Attention},
  author={Marcus},
  journal={arXiv preprint arXiv:2607.XXXXX},
  year={2026}
}
```
