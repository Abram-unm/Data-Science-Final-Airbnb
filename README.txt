First install requirements:
- pip install -r requirements.txt

Second prepare the data for Random Forest:
- cd src
- python preprocess_rf.py

Third run Random Forest script:
- python random_forest.py --data_path ../

Forth run SVM:
- python src/svm.py

Fifth run otimizing price model:
- python src/optimize_price.py
