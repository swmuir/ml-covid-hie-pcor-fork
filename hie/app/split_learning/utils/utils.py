from pathlib import Path

# paths


def project_path() -> Path:
    #print(list(Path(__file__).resolve().parents))
    return Path(__file__).resolve().parents[1]


def project_root_path() -> Path:
    #print(list(project_path().parents))
    return project_path().parents[1]


def workspace_root_path() -> Path:
    #print(list(project_root_path().parents))
    return project_root_path().parents[0]


def config_path() -> Path:
    return workspace_root_path() / "configs"


def data_path() -> Path:
    return workspace_root_path() / "data"


def docs_path() -> Path:
    return workspace_root_path() / "docs"


def model_path() -> Path:
    return project_root_path() / "models"
