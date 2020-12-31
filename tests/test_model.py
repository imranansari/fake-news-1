import numpy as np

from fake_news.model.tree_based import RandomForestModel


def test_rf_overfits_small_dataset():
    model = RandomForestModel()
    train_features = np.random.randn(3, 4)
    train_labels = [True, False, True]
    
    model.train(train_features, train_labels)
    predicted_labels = np.argmax(model.predict(train_features), axis=1)
    predicted_labels = list(map(lambda x: bool(x), predicted_labels))
    assert predicted_labels == train_labels


def test_rf_correct_predict_shape():
    model = RandomForestModel()
    train_features = np.random.randn(3, 4)
    train_labels = [True, False, True]
    
    model.train(train_features, train_labels)
    predicted_labels = np.argmax(model.predict(train_features), axis=1)
    
    assert predicted_labels.shape[0] == 3


def test_rf_correct_predict_range():
    model = RandomForestModel()
    train_features = np.random.randn(3, 4)
    train_labels = [True, False, True]
    
    model.train(train_features, train_labels)
    predicted_probs = model.predict(train_features)
    
    assert (predicted_probs <= 1).all()
