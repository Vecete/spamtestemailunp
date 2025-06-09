import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Carregar dados
# Supondo um DataFrame com colunas: ['body', 'label binário']
df = pd.read_csv('spam_ham_dataset.csv', engine = 'python') 

# Pré-processamento integrado
def preprocess_data(df):
    df = df.copy()
    # Feature engineering
    df['body_length'] = df['body'].apply(len)
    df['num_links'] = df['body'].str.count(r'https?://')
    df['num_exclam'] = df['body'].str.count('!')
    return df

df_processed = preprocess_data(df)

X = df_processed[['body', 'body_length', 'num_links', 'num_exclam']]
y = df_processed['label_num']  # Coluna alvo binária (0: não-spam, 1: spam)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


text_transformer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'body'),
        ('num', StandardScaler(), ['body_length', 'num_links', 'num_exclam'])
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='saga', max_iter=1000, random_state=42))
])

# Busca de hiperparâmetros
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== Melhores hiperparâmetros ===")
print(grid_search.best_params_)

print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred))

print(f"\n=== AUC-ROC: {roc_auc_score(y_test, y_pred):.4f} ===")


if hasattr(best_model.named_steps['classifier'], 'coef_'):
    print("\n=== Features mais importantes ===")
    text_features = best_model.named_steps['preprocessor'].named_transformers_['text'].get_feature_names_out()
    all_features = np.concatenate([
        text_features,
        ['body_length', 'num_links', 'num_exclam']
    ])
    
    coefs = best_model.named_steps['classifier'].coef_[0]
    top_spam = sorted(zip(all_features, coefs), key=lambda x: x[1], reverse=True)[:10]
    top_ham = sorted(zip(all_features, coefs), key=lambda x: x[1])[:10]
    
    print("\nTop 10 para SPAM:")
    print([f"{feat}: {weight:.3f}" for feat, weight in top_spam])
    
    print("\nTop 10 para NÃO-SPAM:")
    print([f"{feat}: {weight:.3f}" for feat, weight in top_ham])