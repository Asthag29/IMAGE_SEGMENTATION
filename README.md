# project_name


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md                 
├── notebooks/                # Jupyter notebooks
├── src/
│   |── tests/                # testing
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_data.py
│   │   └── test_model.py 
│   ├── train/                # training
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   └── train.py
│   ├── data/                 # data
│   │   ├── raw/
│   │   └── processed/
│   ├── models/               # Trained models
│   └──  reports/             # Reports
│       └── figures/
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

