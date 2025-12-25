# Trading Dashboard (Example)

This repo is an example you can publish with GitHub Pages and view on your phone.

## Quick publish (GitHub Pages)
1. Create a new GitHub repo and upload the contents of this folder.
2. In GitHub: Settings → Pages → Deploy from branch → main → /docs
3. Open: https://<your-username>.github.io/<repo-name>/

## Updating after each trade session (Option B)
Keep your raw data locally in `raw/` (gitignored), then run:

```bash
pip install plotly pandas numpy
python analysis/analyze_interactive.py --input-dir raw --output-dir docs
git add docs/*
git commit -m "Update dashboard"
git push
```

Only `docs/` is published; raw files stay private.
