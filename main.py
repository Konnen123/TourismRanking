from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import matplotlib.pyplot as plt
import os


def create_ranking(country, test_size):
    df = pd.read_csv('tourism_dataset.csv')
    df = df[df['Country'] == country]

    df['Revenue_per_visitor'] = df['Revenue'] / df['Visitors']

    X = pd.get_dummies(df[['Category', 'Visitors']], drop_first=False)
    y = df['Revenue_per_visitor']
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    categories = df['Category'].unique()
    predictions = []

    for category in categories:
        input_data = pd.DataFrame({'Category': [category], 'Visitors': [df['Visitors'].mean()]})
        input_data = pd.get_dummies(input_data, drop_first=False)

        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data[feature_names]

        input_data_scaled = scaler.transform(input_data)

        predicted_revenue = knn.predict(input_data_scaled)
        predictions.append((category, predicted_revenue[0]))

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    print(f"Ierarhia categoriilor pentru țara: {country} (test size: {test_size})")
    for rank, (category, revenue_per_visitor) in enumerate(predictions, start=1):
        print(f"{rank}. {category}: {revenue_per_visitor:.2f} ")
    print('\n')

    plt.figure(figsize=(10, 6))
    categories, revenues = zip(*predictions)
    plt.barh(categories, revenues, color='skyblue')
    plt.xlabel('Revenue per Visitor')
    plt.ylabel('Category')
    plt.title(f'Category Hierarchy for {country} with test size {test_size}')
    plt.gca().invert_yaxis()

    folder_name = test_size
    if not os.path.exists(f'figures/{folder_name}'):
        os.makedirs(f'figures/{folder_name}')

    plt.savefig(f'figures/{test_size}/{country}_category_hierarchy.png')


def create_rankings_for_all_countries():
    countries = ['China', 'France', 'Brazil', 'India', 'Egypt', 'USA', 'Australia']
    test_sizes = [0.1, 0.2]

    for country in countries:
        for test_size in test_sizes:
            create_ranking(country, test_size)


if __name__ == '__main__':
    if not os.path.exists('figures'):
        os.makedirs('figures')

    create_rankings_for_all_countries()
