import src.pipeline
print(src.pipeline.__file__)

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
