from session import QFinDASession


def main():

    data_path = r'C:\Users\yixia\OneDrive\Files\Kaggle\jane-street-market-prediction\train.csv'
    session = QFinDASession()
    session.add_data_from_csv(data_path)


if __name__ == '__main__':

    main()
