# Kaggle
Kaggle competition codes and implementation of papers.


## Tips
- Register virtual environment to Jupyter
```bash
poetry add ipykernel
./.venv/bin/ipython kernel install --user --name=kaggle
```

- Add opencv stub
```bash
curl -sSL https://raw.githubusercontent.com/bschnurr/python-type-stubs/add-opencv/cv2/__init__.pyi -o $(poetry run python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')/cv2.pyi
```
