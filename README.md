````markdown
# fagprojekt

KV-Cache-Projekt

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````

# Step-by-step guide til at åbne repo i HPC:
1. Åbn launch.bat
2. Wait for dispatch
3. I terminalen: sh start_tunnel.sh
3. Åben VSCode
4. Connect til tunnel
5. Open folder til projektfolder (inde i ~work3/STUDIENUMMER/Fagprojekt-2026)

**HUSK** NÅR DU ER FÆRDIG:
To close node:
Ctrl + C
Exit

bstat (for at tjekke om andre nodes stadig kører)

Hvis ja:
bkill JOBID 


# To add a new package to the environment, run the following in the terminal:
1. uv add [package name] 
2. git add uv.lock pyproject.toml
3. git commit -m "updated .venv"
4. git push

# To activate .venv
Mac: source .venv/bin/activate

Other: .venv/scripts/activate

# To add content
1. Save file
2. Stage file: git add [filename] (or "git add ." to stage all changed files)
3. Commit file: git commit -m "message"
4. Push file: git push

# To get content
1. git pull
2. uv sync


# To get the data and clean it

1. Make a huggingface API token on: https://huggingface.co/settings/tokens
2. COPY THE API KEY
3. Make a .env file in the root of the repository.
It should contain the following:
HUGGINGFACE_HUB_TOKEN=your huggingface API token
4. Run the data.py file


# Interactive GPUs
Login into Thinlinc and write in terminal:

- 1 interactive V100-node with NVlink reachable via `sxm2sh`
- 1 interactive A100-node with NVlink reachable via `a100sh`

Write:

`code tunnel`

